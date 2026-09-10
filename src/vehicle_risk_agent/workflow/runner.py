"""Assessment workflow runner integrating LangGraph and domain repositories."""

# story: e07s01

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.anthropic_drafting import AnthropicDraftingAdapter
from vehicle_risk_agent.adapters.mcp import VehicleMcpClientAdapter, create_mcp_adapter
from vehicle_risk_agent.api.models import AssessmentContext
from vehicle_risk_agent.config import DEFAULT_SNAPSHOT_INTEGRITY_SECRET, Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceRepository
from vehicle_risk_agent.observability.telemetry import trace_boundary
from vehicle_risk_agent.persistence.event_store import EventStore
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.policy.corpus_lifecycle import CorpusLifecycleManager
from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
from vehicle_risk_agent.reporting.protocol import ReportDraftingProtocol
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
from vehicle_risk_agent.retrieval.adapters import (
    CrossEncoderRerankerAdapter,
    EmbeddingAdapter,
    FakeEmbeddingAdapter,
    FakeRerankerAdapter,
    RerankerAdapter,
    SentenceTransformersEmbeddingAdapter,
)
from vehicle_risk_agent.retrieval.postgres_index import PostgresPolicyIndex
from vehicle_risk_agent.retrieval.service import HybridRetrievalService
from vehicle_risk_agent.risk.repository import RiskPolicyRepository, RiskResultRepository
from vehicle_risk_agent.workflow.graph import build_assessment_graph
from vehicle_risk_agent.workflow.state import AssessmentGraphState


def build_initial_state(
    assessment_id: str,
    run_number: int,
    vin: str,
    context: AssessmentContext,
) -> AssessmentGraphState:
    """Build the initial state for one persisted Assessment Run."""
    return {
        "assessment_id": assessment_id,
        "run_number": run_number,
        "vin": vin,
        "context": context,
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }


class AssessmentWorkflowRunner:
    """Manages workflow execution with PostgreSQL checkpoints and event storage."""

    def __init__(
        self,
        checkpointer: AsyncPostgresSaver,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        broadcaster: ProgressEventBroadcaster | None = None,
        mcp_adapter: VehicleMcpClientAdapter | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
        reranker_adapter: RerankerAdapter | None = None,
        integrity_secret: str = DEFAULT_SNAPSHOT_INTEGRITY_SECRET,
        settings: Settings | None = None,
        drafting_adapter: ReportDraftingProtocol | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.session_factory = session_factory
        self.broadcaster = broadcaster
        self.mcp_adapter = mcp_adapter
        self.embedding_adapter = embedding_adapter
        self.reranker_adapter = reranker_adapter
        self.integrity_secret = integrity_secret
        self.settings = settings
        self.drafting_adapter = drafting_adapter
        self._graph = build_assessment_graph()
        self._app: CompiledStateGraph = self._graph.compile(checkpointer=self.checkpointer)  # type: ignore[type-arg]

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        broadcaster: ProgressEventBroadcaster | None = None,
        mcp_adapter: VehicleMcpClientAdapter | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
        reranker_adapter: RerankerAdapter | None = None,
        drafting_adapter: ReportDraftingProtocol | None = None,
    ) -> AsyncIterator["AssessmentWorkflowRunner"]:
        """Create and initialize a runner with AsyncPostgresSaver and optional session factory."""
        conn_string = settings.database_url.replace("+psycopg", "")
        dispose_engine = False
        engine = None
        if session_factory is None:
            engine = create_async_engine(settings.database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            dispose_engine = True

        configured_mcp_adapter = mcp_adapter or create_mcp_adapter(settings)
        async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
            await checkpointer.setup()
            try:
                yield cls(
                    checkpointer=checkpointer,
                    session_factory=session_factory,
                    broadcaster=broadcaster,
                    mcp_adapter=configured_mcp_adapter,
                    embedding_adapter=embedding_adapter,
                    reranker_adapter=reranker_adapter,
                    integrity_secret=settings.snapshot_integrity_secret.get_secret_value(),
                    settings=settings,
                    drafting_adapter=drafting_adapter,
                )
            finally:
                if dispose_engine and engine is not None:
                    await engine.dispose()

    @property
    def app(self) -> CompiledStateGraph:  # type: ignore[type-arg]
        """Return the compiled workflow application."""
        return self._app

    @staticmethod
    def get_thread_id(assessment_id: str, run_number: int) -> str:
        """Format a stable, bounded thread_id identifier (< 255 chars)."""
        return f"{assessment_id}:{run_number}"

    async def run(
        self,
        initial_state: AssessmentGraphState | None = None,
        assessment_id: str | None = None,
        run_number: int | None = None,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        """Execute the workflow, automatically managing thread_id and EventStore injection."""
        asmt_id = assessment_id or (initial_state["assessment_id"] if initial_state else None)
        run_num = run_number or (initial_state["run_number"] if initial_state else None)
        if not asmt_id or run_num is None:
            raise ValueError("assessment_id and run_number required to determine execution thread")

        thread_id = self.get_thread_id(asmt_id, run_num)
        run_config: dict[str, Any] = dict(config or {})
        configurable = dict(run_config.get("configurable", {}))
        configurable["thread_id"] = thread_id
        if "mcp_adapter" not in configurable and self.mcp_adapter is not None:
            configurable["mcp_adapter"] = self.mcp_adapter

        with trace_boundary(
            "workflow.execute",
            boundary="workflow",
            assessment_id=asmt_id,
            run_number=run_num,
        ):
            # If session_factory is available and event_store not explicitly passed in configurable
            if self.session_factory is not None:
                async with self.session_factory() as session:
                    with trace_boundary(
                        "database.workflow_session",
                        boundary="database",
                        assessment_id=asmt_id,
                        run_number=run_num,
                    ):
                        if "event_store" not in configurable:
                            store = EventStore(session, broadcaster=self.broadcaster)
                            configurable["event_store"] = store
                        if "evidence_repo" not in configurable:
                            configurable["evidence_repo"] = VehicleEvidenceRepository(
                                session,
                                integrity_secret=self.integrity_secret,
                            )
                        if "policy_repo" not in configurable:
                            configurable["policy_repo"] = RiskPolicyRepository(session)
                        if "risk_repo" not in configurable:
                            configurable["risk_repo"] = RiskResultRepository(session)
                        if "draft_repo" not in configurable:
                            configurable["draft_repo"] = ReportDraftRepository(session)
                        if "assessment_repo" not in configurable:
                            configurable["assessment_repo"] = AssessmentRepository(session)
                        if "retrieval_service" not in configurable:
                            active_corpus = await CorpusLifecycleManager(
                                session
                            ).get_active_corpus()
                            if active_corpus is not None:
                                retrieval_mode = (
                                    getattr(self.settings, "retrieval_mode", "offline")
                                    if self.settings is not None
                                    else "offline"
                                )
                                is_live_retrieval = retrieval_mode.lower() == "live"
                                if self.embedding_adapter is not None:
                                    embedder = self.embedding_adapter
                                elif is_live_retrieval:
                                    embedder = SentenceTransformersEmbeddingAdapter(
                                        model_name=active_corpus.retrieval_config.embedding_model,
                                        revision=active_corpus.retrieval_config.embedding_revision,
                                        dimensions=active_corpus.retrieval_config.embedding_dimensions,
                                    )
                                else:
                                    embedder = FakeEmbeddingAdapter(
                                        dimensions=active_corpus.retrieval_config.embedding_dimensions,
                                    )

                                if self.reranker_adapter is not None:
                                    reranker = self.reranker_adapter
                                elif is_live_retrieval:
                                    reranker = CrossEncoderRerankerAdapter(
                                        model_name=active_corpus.retrieval_config.reranker_model,
                                        revision=active_corpus.retrieval_config.reranker_revision,
                                    )
                                else:
                                    reranker = FakeRerankerAdapter()

                                index = PostgresPolicyIndex(
                                    session,
                                    embedder,
                                    active_corpus.snapshot_ids,
                                    active_corpus.retrieval_config,
                                )
                                configurable["retrieval_service"] = HybridRetrievalService(
                                    index,
                                    reranker,
                                    active_corpus.retrieval_config,
                                )

                        if "drafting_adapter" not in configurable:
                            if self.drafting_adapter is not None:
                                configurable["drafting_adapter"] = self.drafting_adapter
                            else:
                                drafting_mode = (
                                    getattr(self.settings, "drafting_mode", "offline").lower()
                                    if self.settings is not None
                                    else "offline"
                                )
                                if drafting_mode == "live":
                                    api_key = (
                                        self.settings.anthropic_api_key.get_secret_value()
                                        if self.settings and self.settings.anthropic_api_key
                                        else None
                                    )
                                    configurable["drafting_adapter"] = AnthropicDraftingAdapter(
                                        model=(
                                            self.settings.drafting_model
                                            if self.settings
                                            else "claude-sonnet-4-6"
                                        ),
                                        max_tokens=(
                                            self.settings.drafting_max_tokens
                                            if self.settings
                                            else 2048
                                        ),
                                        timeout_seconds=(
                                            int(self.settings.drafting_timeout_seconds)
                                            if self.settings
                                            else 30
                                        ),
                                        api_key=api_key,
                                    )
                                else:
                                    drafter = OfflineReportDraftingAdapter()
                                    configurable["drafting_adapter"] = drafter

                        run_config["configurable"] = configurable
                        assessment_repo = configurable.get("assessment_repo")
                        if assessment_repo is not None:
                            await assessment_repo.ensure_run(asmt_id, run_num)
                        try:
                            result: dict[str, Any] = await self._app.ainvoke(
                                initial_state, config=cast(RunnableConfig, run_config)
                            )
                            phase_value = result.get("phase", AssessmentRunPhase.FAILED)
                            try:
                                final_phase = AssessmentRunPhase(phase_value)
                            except (TypeError, ValueError):
                                final_phase = AssessmentRunPhase.FAILED
                            if assessment_repo is not None:
                                await assessment_repo.update_run_phase(
                                    asmt_id, run_num, final_phase
                                )
                            else:
                                await session.commit()
                            return result
                        except Exception:
                            await session.rollback()
                            if assessment_repo is not None:
                                try:
                                    await assessment_repo.update_run_phase(
                                        asmt_id, run_num, AssessmentRunPhase.FAILED
                                    )
                                except Exception:
                                    await session.rollback()
                            raise

            run_config["configurable"] = configurable
            res: dict[str, Any] = await self._app.ainvoke(
                initial_state, config=cast(RunnableConfig, run_config)
            )
            return res


class AssessmentRunner:
    """Lightweight in-memory or integration runner for assessment graph execution."""

    def __init__(
        self,
        mcp_adapter: VehicleMcpClientAdapter | None = None,
        evidence_repo: VehicleEvidenceRepository | None = None,
    ) -> None:
        self.mcp_adapter = mcp_adapter
        self.evidence_repo = evidence_repo
        self._graph = build_assessment_graph()
        self._app: CompiledStateGraph = self._graph.compile()  # type: ignore[type-arg]

    async def run(
        self,
        assessment_id: str,
        run_number: int,
        vin: str,
        context: AssessmentContext,
    ) -> dict[str, Any]:
        """Execute the assessment graph without checkpointer."""
        initial_state = build_initial_state(assessment_id, run_number, vin, context)
        config: dict[str, Any] = {
            "configurable": {
                "mcp_adapter": self.mcp_adapter,
                "evidence_repo": self.evidence_repo,
            }
        }
        with trace_boundary(
            "workflow.execute",
            boundary="workflow",
            assessment_id=assessment_id,
            run_number=run_number,
        ):
            res: dict[str, Any] = await self._app.ainvoke(
                initial_state, config=cast(RunnableConfig, config)
            )
            return res

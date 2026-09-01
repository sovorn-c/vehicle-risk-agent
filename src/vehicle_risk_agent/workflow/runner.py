"""Assessment workflow runner integrating LangGraph and domain repositories."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.events.broadcaster import ProgressEventBroadcaster
from vehicle_risk_agent.persistence.event_store import EventStore
from vehicle_risk_agent.workflow.graph import build_assessment_graph
from vehicle_risk_agent.workflow.state import AssessmentGraphState


class AssessmentWorkflowRunner:
    """Manages workflow execution with PostgreSQL checkpoints and event storage."""

    def __init__(
        self,
        checkpointer: AsyncPostgresSaver,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        broadcaster: ProgressEventBroadcaster | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.session_factory = session_factory
        self.broadcaster = broadcaster
        self._graph = build_assessment_graph()
        self._app: CompiledStateGraph = self._graph.compile(checkpointer=self.checkpointer)  # type: ignore[type-arg]

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        broadcaster: ProgressEventBroadcaster | None = None,
    ) -> AsyncIterator["AssessmentWorkflowRunner"]:
        """Create and initialize a runner with AsyncPostgresSaver and optional session factory."""
        conn_string = settings.database_url.replace("+psycopg", "")
        dispose_engine = False
        engine = None
        if session_factory is None:
            engine = create_async_engine(settings.database_url, echo=False)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            dispose_engine = True

        async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
            await checkpointer.setup()
            try:
                yield cls(
                    checkpointer=checkpointer,
                    session_factory=session_factory,
                    broadcaster=broadcaster,
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

        # If session_factory is available and event_store not explicitly passed in configurable
        if "event_store" not in configurable and self.session_factory is not None:
            async with self.session_factory() as session:
                store = EventStore(session, broadcaster=self.broadcaster)
                configurable["event_store"] = store
                run_config["configurable"] = configurable
                result: dict[str, Any] = await self._app.ainvoke(
                    initial_state, config=cast(RunnableConfig, run_config)
                )
                await session.commit()
                return result

        run_config["configurable"] = configurable
        res: dict[str, Any] = await self._app.ainvoke(
            initial_state, config=cast(RunnableConfig, run_config)
        )
        return res

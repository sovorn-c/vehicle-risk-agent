"""Adversarial stress-testing harness for LangGraph assessment workflow graph & runner.

Empirical Challenger 2 Test Suite for Milestone 2 (e04s02).
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import (
    FakeVehicleMcpAdapter,
    McpAdapterError,
    VehicleMcpClientAdapter,
)
from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    ProvenanceLink,
    SafeError,
    SafeErrorCategory,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.reporting.models import (
    ReportDraft,
    ReportDraftStatus,
)
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskResult,
    build_risk_policy_v1,
)
from vehicle_risk_agent.risk.repository import RiskPolicyRepository
from vehicle_risk_agent.risk.service import RiskPolicyService
from vehicle_risk_agent.workflow.graph import (
    build_assessment_graph,
    node_drafting_report,
)
from vehicle_risk_agent.workflow.runner import AssessmentRunner, AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import (
    AssessmentGraphState,
    reduce_events,
    reduce_visited_phases,
)

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"
VALID_VIN_1 = "1HGCR2F85HA000000"
VALID_VIN_2 = "7AT0BJ03X20000001"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Provide a fresh database session with all tables created."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        yield sess

    await engine.dispose()


def _build_complete_vehicle_revision(
    vin: str = VALID_VIN_1,
    ppsr: str = "NO_FINANCE_REGISTERED",
    stolen: str = "NOT_STOLEN",
    writeoff: str = "NOT_WRITTEN_OFF",
    revision_number: int = 1,
) -> VehicleRevisionResponse:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    return VehicleRevisionResponse(
        vin=vin,
        revision_id=f"rev-{vin}-{revision_number}",
        revision_number=revision_number,
        material_hash="a" * 64,
        canonical_fields={
            "vin": vin,
            "make": "HONDA",
            "model": "ACCORD",
            "year": 2017,
            "plate": "NZACC1",
            "ppsr_result": ppsr,
            "stolen_status": stolen,
            "writeoff_status": writeoff,
        },
        field_provenance={
            "vin": [
                ProvenanceLink(
                    observation_id="obs-vin",
                    source_system="NZTA",
                    source_record_id="rec-vin",
                    retrieved_at=now,
                )
            ],
            "ppsr_result": [
                ProvenanceLink(
                    observation_id="obs-ppsr",
                    source_system="PPSR",
                    source_record_id="rec-ppsr",
                    retrieved_at=now,
                )
            ],
            "stolen_status": [
                ProvenanceLink(
                    observation_id="obs-police",
                    source_system="POLICE",
                    source_record_id="rec-police",
                    retrieved_at=now,
                )
            ],
            "writeoff_status": [
                ProvenanceLink(
                    observation_id="obs-writeoff",
                    source_system="NZTA",
                    source_record_id="rec-writeoff",
                    retrieved_at=now,
                )
            ],
        },
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )


# =============================================================================
# 1. FULL PIPELINE EXECUTION & MULTI-RISK STRESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_full_pipeline_multi_risk_evaluation_and_draft_persistence(
    session: AsyncSession,
) -> None:
    """Full workflow execution with multiple active risk factors (PPSR + Stolen + Writeoff)."""
    # 1. Seed assessment & active policy
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="challenger-2",
        idempotency_key="idemp-multi-risk-1",
        request=AssessmentCreateRequest(
            vin=VALID_VIN_1,
            context=AssessmentContext(sale_type=SaleType.DEALER, intended_use="Commercial"),
        ),
    )
    assert asmt.lifecycle_state == AssessmentLifecycleState.IN_PROGRESS

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    # 2. Seed vehicle with multiple high-risk flags
    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_complete_vehicle_revision(
        vin=VALID_VIN_1,
        ppsr="MATCH",
        stolen="STOLEN",
        writeoff="STATUTORY",
    )
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": VALID_VIN_1,
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.run(initial_state=initial_state)

        # 3. Verify final state and visited phase sequence
        assert result["phase"] == AssessmentRunPhase.COMPLETED
        expected_phases = [
            AssessmentRunPhase.PENDING,
            AssessmentRunPhase.COLLECTING_EVIDENCE,
            AssessmentRunPhase.EVALUATING_SUFFICIENCY,
            AssessmentRunPhase.RETRIEVING_POLICY,
            AssessmentRunPhase.EVALUATING_RISK,
            AssessmentRunPhase.DRAFTING_REPORT,
            AssessmentRunPhase.COMPLETED,
        ]
        assert result["visited_phases"] == expected_phases

        # 4. Verify risk result
        risk_result: RiskResult = result["risk_result"]
        assert risk_result is not None
        assert risk_result.is_incomplete is False
        assert risk_result.score == 100  # Capped at 100
        assert risk_result.band == RiskBand.CRITICAL
        assert len(risk_result.findings) >= 3

        # 5. Verify report draft
        draft: ReportDraft = result["report_draft"]
        assert draft is not None
        assert draft.outcome == AssessmentOutcome.SCORED
        assert draft.score == 100
        assert draft.band == RiskBand.CRITICAL
        assert draft.status == ReportDraftStatus.DRAFT
        assert len(draft.sections.as_list()) == 9
        assert len(draft.sections.mandatory_review_findings.findings) >= 3
        assert draft.sections.executive_summary.recommendation != ""

        # 6. Verify database persistence and aggregate transition
        draft_repo = ReportDraftRepository(session)
        stored_draft = await draft_repo.get_draft(asmt.id, 1)
        assert stored_draft is not None
        assert stored_draft.draft_hash == draft.draft_hash
        assert stored_draft.score == 100
        assert stored_draft.band == RiskBand.CRITICAL

        updated_asmt = await asmt_repo.get_assessment(asmt.id)
        assert updated_asmt is not None
        assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW

    await engine.dispose()


# =============================================================================
# 2. INCOMPLETE NODE ROUTING & WITHHELD SCORING ADVERSARIAL TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_routing_incomplete_missing_stolen_and_writeoff(
    session: AsyncSession,
) -> None:
    """Missing stolen_status and writeoff_status routes to incomplete and withholds score."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="challenger-2",
        idempotency_key="idemp-inc-2",
        request=AssessmentCreateRequest(
            vin=VALID_VIN_2,
            context=AssessmentContext(sale_type=SaleType.PRIVATE),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    # Seed vehicle with missing stolen and writeoff fields
    mcp_adapter = FakeVehicleMcpAdapter()
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin=VALID_VIN_2,
        revision_id="rev-inc-2",
        revision_number=1,
        material_hash="b" * 64,
        canonical_fields={
            "vin": VALID_VIN_2,
            "make": "MAZDA",
            "model": "AXELA",
            "year": 2018,
            "ppsr_result": "NO_FINANCE_REGISTERED",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=50,
            band=ConfidenceBand.LOW,
            rule_version="v1",
            explanation="partial missing fields",
        ),
        as_of=now,
        published_at=now,
    )
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": VALID_VIN_2,
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.run(initial_state=initial_state)

        # Verify phase is INCOMPLETE
        assert result["phase"] == AssessmentRunPhase.INCOMPLETE
        expected_phases = [
            AssessmentRunPhase.PENDING,
            AssessmentRunPhase.COLLECTING_EVIDENCE,
            AssessmentRunPhase.EVALUATING_SUFFICIENCY,
            AssessmentRunPhase.INCOMPLETE,
        ]
        assert result["visited_phases"] == expected_phases

        # Verify draft was generated with withheld score
        draft: ReportDraft = result["report_draft"]
        assert draft.outcome == AssessmentOutcome.INCOMPLETE
        assert draft.score is None
        assert draft.band is None
        assert draft.is_incomplete is True
        assert len(draft.missing_evidence_notices) == 2
        missing_names = [m.field_name for m in draft.missing_evidence_notices]
        assert "stolen_status" in missing_names
        assert "writeoff_status" in missing_names

        # Verify safe event message captures exact missing count
        events: list[WorkflowProgressEvent] = result["events"]
        incomplete_event = next(e for e in events if e.phase == AssessmentRunPhase.INCOMPLETE)
        assert (
            "withheld scoring due to 2 incomplete required evidence fields"
            in incomplete_event.safe_message
        )

        # Verify DB persistence of incomplete draft & assessment transition
        draft_repo = ReportDraftRepository(session)
        stored_draft = await draft_repo.get_draft(asmt.id, 1)
        assert stored_draft is not None
        assert stored_draft.outcome == AssessmentOutcome.INCOMPLETE
        assert stored_draft.score is None

        updated_asmt = await asmt_repo.get_assessment(asmt.id)
        assert updated_asmt is not None
        assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW

    await engine.dispose()


@pytest.mark.asyncio
async def test_workflow_routing_incomplete_unresolved_conflicts(
    session: AsyncSession,
) -> None:
    """Unresolved conflict marks sufficiency as incomplete and routes to incomplete node."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="challenger-2",
        idempotency_key="idemp-conflict-1",
        request=AssessmentCreateRequest(
            vin=VALID_VIN_1,
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    conflicts = [
        FieldConflict(
            field_name="ppsr_result",
            conflicting_candidates=[
                CandidateValue(
                    field_name="ppsr_result",
                    value="MATCH",
                    provenance=ProvenanceLink(
                        observation_id="obs-ppsr-1",
                        source_system="PPSR_REG",
                        source_record_id="rec-1",
                        retrieved_at=now,
                    ),
                ),
                CandidateValue(
                    field_name="ppsr_result",
                    value="NO_FINANCE_REGISTERED",
                    provenance=ProvenanceLink(
                        observation_id="obs-ppsr-2",
                        source_system="PPSR_REG",
                        source_record_id="rec-2",
                        retrieved_at=now,
                    ),
                ),
            ],
            state=ConflictState.UNRESOLVED,
            winning_value=None,
            rule_version="v1",
            rationale="Unresolvable conflict between PPSR records",
        )
    ]

    rev = VehicleRevisionResponse(
        vin=VALID_VIN_1,
        revision_id="rev-conflict-1",
        revision_number=1,
        material_hash="c" * 64,
        canonical_fields={
            "vin": VALID_VIN_1,
            "make": "TOYOTA",
            "model": "COROLLA",
            "year": 2020,
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=conflicts,
        confidence=ConfidenceAssessment(
            score=40,
            band=ConfidenceBand.LOW,
            rule_version="v1",
            explanation="Unresolved conflict",
        ),
        as_of=now,
        published_at=now,
    )

    mcp_adapter = FakeVehicleMcpAdapter()
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": VALID_VIN_1,
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.run(initial_state=initial_state)

        assert result["phase"] == AssessmentRunPhase.INCOMPLETE
        draft: ReportDraft = result["report_draft"]
        assert draft.outcome == AssessmentOutcome.INCOMPLETE
        assert draft.score is None
        assert len(draft.missing_evidence_notices) >= 1
        assert draft.missing_evidence_notices[0].field_name == "ppsr_result"
        assert "UNRESOLVED_CONFLICT" in draft.missing_evidence_notices[0].reason

    await engine.dispose()


# =============================================================================
# 3. REPOSITORY INJECTION FAILURE & PURE IN-MEMORY FALLBACK TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_pure_in_memory_execution_without_db_or_repos() -> None:
    """Graph executes in pure memory when session_factory is None and no repos are configured."""
    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_complete_vehicle_revision(VALID_VIN_1)
    mcp_adapter.seed_vehicle(rev)

    # Use standalone AssessmentRunner (pure in-memory)
    runner = AssessmentRunner(mcp_adapter=mcp_adapter)
    result = await runner.run(
        assessment_id="asmt-mem-01",
        run_number=1,
        vin=VALID_VIN_1,
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )

    assert result["phase"] == AssessmentRunPhase.COMPLETED
    assert result.get("evidence_snapshot") is not None
    assert result.get("risk_result") is not None
    assert result.get("report_draft") is not None
    draft: ReportDraft = result["report_draft"]
    assert draft.outcome == AssessmentOutcome.SCORED
    assert draft.score == 0
    assert len(draft.sections.as_list()) == 9


@pytest.mark.asyncio
async def test_workflow_incomplete_pure_in_memory_fallback() -> None:
    """Incomplete evidence path completes cleanly in pure in-memory mode without repos."""
    mcp_adapter = FakeVehicleMcpAdapter()
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    rev = VehicleRevisionResponse(
        vin=VALID_VIN_2,
        revision_id="rev-mem-inc",
        revision_number=1,
        material_hash="d" * 64,
        canonical_fields={
            "vin": VALID_VIN_2,
            "make": "HONDA",
            "model": "FIT",
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=30,
            band=ConfidenceBand.LOW,
            rule_version="v1",
            explanation="missing",
        ),
        as_of=now,
        published_at=now,
    )
    mcp_adapter.seed_vehicle(rev)

    runner = AssessmentRunner(mcp_adapter=mcp_adapter)
    result = await runner.run(
        assessment_id="asmt-mem-inc-01",
        run_number=1,
        vin=VALID_VIN_2,
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )

    assert result["phase"] == AssessmentRunPhase.INCOMPLETE
    assert result.get("report_draft") is not None
    draft: ReportDraft = result["report_draft"]
    assert draft.outcome == AssessmentOutcome.INCOMPLETE
    assert draft.score is None


# =============================================================================
# 4. ERROR ROUTING & SENSITIVE DATA MINIMIZATION STRESS TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_routing_failed_when_mcp_adapter_missing() -> None:
    """Missing MCP adapter routes to failed node with retryable PIPELINE_UNAVAILABLE SafeError."""
    graph = build_assessment_graph()
    app = graph.compile()

    initial_state: AssessmentGraphState = {
        "assessment_id": "asmt-no-mcp",
        "run_number": 1,
        "vin": VALID_VIN_1,
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    result = await app.ainvoke(initial_state, config={"configurable": {}})

    assert result["phase"] == AssessmentRunPhase.FAILED
    assert result.get("mcp_error") is not None
    err: SafeError = result["mcp_error"]
    assert err.category == SafeErrorCategory.PIPELINE_UNAVAILABLE
    assert err.retryable is True
    assert "MCP adapter is not configured" in err.message
    assert result.get("report_draft") is None


class BrokenMcpAdapter(VehicleMcpClientAdapter):
    """Mcp adapter that raises controlled or unexpected exceptions."""

    def __init__(self, error_to_raise: Exception) -> None:
        self.error_to_raise = error_to_raise

    async def lookup_vehicle(self, vin: str) -> VehicleRevisionResponse:
        del vin
        raise self.error_to_raise

    async def explain_vehicle_field(self, vin: str, field_name: str) -> Any:
        del vin, field_name
        raise self.error_to_raise

    async def get_vehicle_revision(self, vin: str, revision_number: int) -> Any:
        del vin, revision_number
        raise self.error_to_raise

    async def get_source_observation(self, observation_id: str) -> Any:
        del observation_id
        raise self.error_to_raise

    async def get_vehicle_history(
        self, vin: str, limit: int = 20, before_revision: int | None = None
    ) -> Any:
        del vin, limit, before_revision
        raise self.error_to_raise


@pytest.mark.asyncio
async def test_workflow_routing_failed_sanitizes_unexpected_exception() -> None:
    """Unexpected internal exceptions are sanitized into generic SafeError."""
    leak_string = "FATAL: Connection failed to postgres://admin:secret@db.internal:5432/prod"
    broken_adapter = BrokenMcpAdapter(RuntimeError(leak_string))

    graph = build_assessment_graph()
    app = graph.compile()

    initial_state: AssessmentGraphState = {
        "assessment_id": "asmt-leak-test",
        "run_number": 1,
        "vin": VALID_VIN_1,
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    result = await app.ainvoke(
        initial_state,
        config={"configurable": {"mcp_adapter": broken_adapter}},
    )

    assert result["phase"] == AssessmentRunPhase.FAILED
    err: SafeError = result["mcp_error"]
    assert err.category == SafeErrorCategory.INTERNAL_ERROR
    assert "secret" not in err.message
    assert "postgres://" not in err.message
    assert err.message == "Unexpected error collecting vehicle evidence"

    events: list[WorkflowProgressEvent] = result["events"]
    for evt in events:
        assert "secret" not in evt.safe_message
        assert "postgres://" not in evt.safe_message


@pytest.mark.asyncio
async def test_workflow_routing_failed_preserves_mcp_adapter_error_category() -> None:
    """Known McpAdapterError preserves its category, retryable flag, and remediation."""
    adapter_err = McpAdapterError(
        category=SafeErrorCategory.VEHICLE_NOT_FOUND,
        message="Vehicle record not found in register",
        retryable=False,
        remediation="Check VIN format.",
    )
    broken_adapter = BrokenMcpAdapter(adapter_err)

    graph = build_assessment_graph()
    app = graph.compile()

    initial_state: AssessmentGraphState = {
        "assessment_id": "asmt-not-found",
        "run_number": 1,
        "vin": VALID_VIN_1,
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }

    result = await app.ainvoke(
        initial_state,
        config={"configurable": {"mcp_adapter": broken_adapter}},
    )

    assert result["phase"] == AssessmentRunPhase.FAILED
    err: SafeError = result["mcp_error"]
    assert err.category == SafeErrorCategory.VEHICLE_NOT_FOUND
    assert err.retryable is False
    assert err.message == "No vehicle evidence was found."


# =============================================================================
# 5. DETERMINISTIC REDUCERS & STATE CONCURRENCY STRESS TESTS
# =============================================================================


def test_reduce_visited_phases_under_adversarial_patterns() -> None:
    """Stress-test reduce_visited_phases with duplicates, repeated loops, and empty inputs."""
    # 1. Empty current, empty new
    assert reduce_visited_phases([], []) == []

    # 2. Consecutive duplicates in new
    p_pending = AssessmentRunPhase.PENDING
    p_collect = AssessmentRunPhase.COLLECTING_EVIDENCE
    assert reduce_visited_phases([p_pending], [p_pending, p_pending, p_collect, p_collect]) == [
        p_pending,
        p_collect,
    ]

    # 3. Legitimate re-visitation (non-consecutive) preserved
    p_eval = AssessmentRunPhase.EVALUATING_SUFFICIENCY
    p_comp = AssessmentRunPhase.COMPLETED
    assert reduce_visited_phases([p_pending, p_collect], [p_eval, p_collect, p_comp]) == [
        p_pending,
        p_collect,
        p_eval,
        p_collect,
        p_comp,
    ]


def test_reduce_events_deduplication_and_sequence_integrity() -> None:
    """Stress-test reduce_events with duplicate and collision keys."""
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    e1 = WorkflowProgressEvent(
        event_id="e1",
        sequence=1,
        assessment_id="asmt-1",
        run_number=1,
        phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
        safe_message="collecting",
        timestamp=now,
    )
    e2 = WorkflowProgressEvent(
        event_id="e2",
        sequence=2,
        assessment_id="asmt-1",
        run_number=1,
        phase=AssessmentRunPhase.EVALUATING_SUFFICIENCY,
        safe_message="evaluating",
        timestamp=now,
    )
    e1_dup = WorkflowProgressEvent(
        event_id="e1-dup",
        sequence=1,
        assessment_id="asmt-1",
        run_number=1,
        phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
        safe_message="collecting duplicate",
        timestamp=now,
    )

    combined = reduce_events([e1], [e1_dup, e2])
    assert len(combined) == 2
    assert combined[0].event_id == "e1"  # original preserved
    assert combined[1].event_id == "e2"


# =============================================================================
# 6. RUNNER THREAD ID BOUNDS & VALIDATION TESTS
# =============================================================================


def test_runner_thread_id_format_and_bounds() -> None:
    """Verify thread ID formatting and length bounds."""
    thread_id = AssessmentWorkflowRunner.get_thread_id("asmt-uuid-12345", 1)
    assert thread_id == "asmt-uuid-12345:1"
    assert len(thread_id) < 255


@pytest.mark.asyncio
async def test_runner_requires_assessment_id_and_run_number() -> None:
    """AssessmentWorkflowRunner.run raises ValueError if assessment_id or run_number is missing."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
    ) as runner:
        with pytest.raises(ValueError, match="assessment_id and run_number required"):
            await runner.run(initial_state={})

    await engine.dispose()


# =============================================================================
# 7. VEHICLE HISTORY, MULTI-RUN CHECKPOINTS & CITATION PROPAGATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_workflow_collects_full_vehicle_history_for_multi_revision(
    session: AsyncSession,
) -> None:
    """Vehicles with revision_number > 1 trigger history collection and store in snapshot."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="challenger-2",
        idempotency_key="idemp-hist-1",
        request=AssessmentCreateRequest(
            vin=VALID_VIN_1,
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    rev2 = _build_complete_vehicle_revision(vin=VALID_VIN_1, revision_number=2)
    rev1 = _build_complete_vehicle_revision(vin=VALID_VIN_1, revision_number=1)

    mcp_adapter = FakeVehicleMcpAdapter()
    mcp_adapter.seed_vehicle(rev1)
    mcp_adapter.seed_vehicle(rev2)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": VALID_VIN_1,
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }

        result = await runner.run(initial_state=initial_state)

        assert result["phase"] == AssessmentRunPhase.COMPLETED
        snapshot = result["evidence_snapshot"]
        assert snapshot is not None
        assert snapshot.revision_number == 2
        assert len(snapshot.history) >= 1
        assert snapshot.history[0].revision_number == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_workflow_multi_run_isolation_on_same_assessment(
    session: AsyncSession,
) -> None:
    """Run 1 and Run 2 for the same assessment_id maintain separate thread_ids and drafts."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="challenger-2",
        idempotency_key="idemp-multirun-1",
        request=AssessmentCreateRequest(
            vin=VALID_VIN_1,
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    # Run 1: clean vehicle
    rev_clean = _build_complete_vehicle_revision(vin=VALID_VIN_1, ppsr="NO_FINANCE_REGISTERED")
    mcp_adapter = FakeVehicleMcpAdapter()
    mcp_adapter.seed_vehicle(rev_clean)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        # Run 1
        state_run_1: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": VALID_VIN_1,
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }
        res1 = await runner.run(initial_state=state_run_1)
        assert res1["phase"] == AssessmentRunPhase.COMPLETED
        assert res1["report_draft"].score == 0

        # Update vehicle in MCP to have PPSR match
        rev_match = _build_complete_vehicle_revision(
            vin=VALID_VIN_1, ppsr="MATCH", revision_number=2
        )
        mcp_adapter.seed_vehicle(rev_match)

        # Run 2
        state_run_2: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 2,
            "vin": VALID_VIN_1,
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }
        res2 = await runner.run(initial_state=state_run_2)
        assert res2["phase"] == AssessmentRunPhase.COMPLETED
        assert res2["report_draft"].score == 30

        # Verify DB holds both runs independently
        draft_repo = ReportDraftRepository(session)
        draft1 = await draft_repo.get_draft(asmt.id, 1)
        draft2 = await draft_repo.get_draft(asmt.id, 2)

        assert draft1 is not None
        assert draft2 is not None
        assert draft1.score == 0
        assert draft2.score == 30
        assert draft1.draft_hash != draft2.draft_hash

    await engine.dispose()


@pytest.mark.asyncio
async def test_workflow_drafting_report_guard_raises_if_risk_result_missing() -> None:
    """node_drafting_report explicitly raises ValueError if state is missing risk_result."""
    state: AssessmentGraphState = {
        "assessment_id": "asmt-guard-01",
        "run_number": 1,
        "vin": VALID_VIN_1,
        "context": AssessmentContext(sale_type=SaleType.DEALER),
        "phase": AssessmentRunPhase.EVALUATING_RISK,
        "visited_phases": [AssessmentRunPhase.EVALUATING_RISK],
        "events": [],
        "risk_result": None,  # Intentionally missing!
    }

    with pytest.raises(ValueError, match="Cannot draft report without risk_result"):
        await node_drafting_report(state)

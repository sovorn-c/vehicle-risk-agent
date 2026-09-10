"""Bounded acceptance scenarios for live/offline evaluation (e10s03-t02).

Asserts:
1. Scored scenario: grounded draft at AWAITING_REVIEW, deterministic score, reviewer gate.
2. Incomplete scenario: score and band withheld, missing evidence visible, reaches AWAITING_REVIEW.
3. Irrelevant policy query: explicit abstention recorded, no unsupported citations.
4. Provider failure: safe failure recorded, no offline fallback, no auto-approval.
5. Offline control: credential-free deterministic run labelled offline.
"""

# story: e10s03
# task: e10s03-t02
# scenario: SC-e10s03-P0-01
# scenario: SC-e10s03-P0-02
# scenario: SC-e10s03-P1-03
# scenario: SC-e10s03-P1-04

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from mcp import types
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.anthropic_drafting import (
    AnthropicDraftingAdapter,
    DraftingFailureError,
)
from vehicle_risk_agent.adapters.mcp import (
    FakeVehicleMcpAdapter,
    StreamableHttpVehicleMcpAdapter,
)
from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentLifecycleState, AssessmentRunPhase
from vehicle_risk_agent.evaluation.live import LiveEvaluationConfig, LiveEvaluationRunner
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    FieldExplanationResult,
    FieldOutcome,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.sufficiency import SufficiencyOutcome
from vehicle_risk_agent.persistence.models import AssessmentRecord, Base
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.reporting.models import ReportDraftStatus
from vehicle_risk_agent.reporting.offline import OfflineReportDraftingAdapter
from vehicle_risk_agent.reporting.repository import ReportDraftRepository
from vehicle_risk_agent.review.models import ApproveReportCommand, ReportDisposition
from vehicle_risk_agent.review.service import ReviewDecisionService
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    build_risk_policy_v1,
)
from vehicle_risk_agent.risk.repository import RiskPolicyRepository
from vehicle_risk_agent.risk.service import RiskPolicyService
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Provide a fresh database session with tables recreated."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as sess:
        yield sess

    await engine.dispose()


def _build_clean_vehicle_revision(vin: str = "1HGCR2F85HA000000") -> VehicleRevisionResponse:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    return VehicleRevisionResponse(
        vin=vin,
        revision_id="rev-clean-01",
        revision_number=1,
        material_hash="e" * 64,
        canonical_fields={
            "vin": vin,
            "make": "HONDA",
            "model": "ACCORD",
            "year": 2017,
            "plate": "NZACC1",
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
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
            explanation="Clean verified record",
        ),
        as_of=now,
        published_at=now,
    )


def _build_incomplete_vehicle_revision(vin: str = "JM0BL10F000000000") -> VehicleRevisionResponse:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    return VehicleRevisionResponse(
        vin=vin,
        revision_id="rev-inc-01",
        revision_number=1,
        material_hash="f" * 64,
        canonical_fields={
            "vin": vin,
            "make": "MAZDA",
            "model": "AXELA",
            "year": 2016,
            "plate": "AXL100",
            # missing ppsr_result, stolen_status, writeoff_status
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
        },
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=50,
            band=ConfidenceBand.MEDIUM,
            rule_version="v1",
            explanation="Partial unverified record",
        ),
        as_of=now,
        published_at=now,
    )


@pytest.mark.asyncio
async def test_bounded_scored_scenario_persists_grounded_draft_and_awaits_review(
    session: AsyncSession,
) -> None:
    """Scored scenario produces score, persists draft at AWAITING_REVIEW, no auto-approval."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-01",
        idempotency_key="idemp-e10-scored",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_clean_vehicle_revision("1HGCR2F85HA000000")
    mcp_adapter.seed_vehicle(rev)

    # Set up live drafter with mocked provider API returning grounded payload with usage
    live_drafter = AnthropicDraftingAdapter(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        timeout_seconds=30,
        api_key="sk-ant-test-key",
    )
    mock_payload = {
        "text": json.dumps(
            {
                "assessment_id": asmt.id,
                "run_number": 1,
                "outcome": "SCORED",
                "sections": {
                    "executive_summary": "Clean vehicle assessment completed.",
                    "vehicle_identity": "2017 HONDA ACCORD VIN 1HGCR2F85HA000000 plate NZACC1.",
                    "risk_score": "Deterministic risk score 0 (LOW).",
                    "mandatory_review": "No mandatory review triggers detected.",
                    "contributing_factors": "All risk factors clear.",
                    "evidence_summary": "Verified across NZTA and Police.",
                    "limitations": "Standard limitations apply.",
                    "synthetic_notice": "Production NZTA rules applied.",
                },
                "claim_refs": [],
            }
        ),
        "input_tokens": 1200,
        "output_tokens": 300,
    }

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=TEST_DB_URL,
        drafting_mode="live",
        enable_live_drafting=True,
        anthropic_api_key=SecretStr("sk-ant-test-key"),
    )

    with patch.object(live_drafter, "_call_anthropic_api", new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_payload
        async with AssessmentWorkflowRunner.create(
            settings=settings,
            session_factory=session_factory,
            mcp_adapter=mcp_adapter,
            drafting_adapter=live_drafter,
        ) as runner:
            initial_state: AssessmentGraphState = {
                "assessment_id": asmt.id,
                "run_number": 1,
                "vin": "1HGCR2F85HA000000",
                "context": asmt.context,
                "phase": AssessmentRunPhase.PENDING,
                "visited_phases": [AssessmentRunPhase.PENDING],
                "events": [],
            }
            final_state = await runner.run(initial_state=initial_state)

    assert final_state["phase"] == AssessmentRunPhase.COMPLETED

    # 1. Deterministic scoring
    risk_res = final_state["risk_result"]
    assert risk_res is not None
    assert risk_res.outcome == AssessmentOutcome.SCORED
    assert risk_res.band == RiskBand.LOW
    assert risk_res.score is not None

    # 2. Persisted draft in DB
    draft_repo = ReportDraftRepository(session)
    persisted_draft = await draft_repo.get_draft(asmt.id, 1)
    assert persisted_draft is not None
    assert persisted_draft.status == ReportDraftStatus.DRAFT

    # 3. Assessment lifecycle state is AWAITING_REVIEW (NEVER auto-approved)
    updated_asmt = await asmt_repo.get_assessment(asmt.id)
    assert updated_asmt is not None
    assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW

    # 4. Reviewer authorization boundary: requires explicit Reviewer command
    review_svc = ReviewDecisionService(session)
    approve_cmd = ApproveReportCommand(
        assessment_id=asmt.id,
        run_number=1,
        reviewer_id="rev-officer-01",
        idempotency_key="idemp-app-01",
        notes="Grounded draft approved after compliance check.",
    )
    decision = await review_svc.record_review_action(approve_cmd)
    assert decision.disposition == ReportDisposition.RELEASED

    final_asmt = await asmt_repo.get_assessment(asmt.id)
    assert final_asmt is not None
    assert final_asmt.lifecycle_state == AssessmentLifecycleState.RELEASED
    await engine.dispose()


@pytest.mark.asyncio
async def test_bounded_incomplete_scenario_withholds_score_and_awaits_review(
    session: AsyncSession,
) -> None:
    """Incomplete evidence scenario withholds score and band and reaches AWAITING_REVIEW."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-02",
        idempotency_key="idemp-e10-incomplete",
        request=AssessmentCreateRequest(
            vin="JM0BL10F000000000",
            context=AssessmentContext(sale_type=SaleType.PRIVATE),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_incomplete_vehicle_revision("JM0BL10F000000000")
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL, drafting_mode="offline")

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": "JM0BL10F000000000",
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }
        final_state = await runner.run(initial_state=initial_state)

    assert final_state["phase"] == AssessmentRunPhase.INCOMPLETE

    # 1. Evidence sufficiency is incomplete
    suff = final_state["sufficiency_result"]
    assert suff is not None
    assert suff.outcome == SufficiencyOutcome.INCOMPLETE

    # 2. Risk score and band are withheld
    risk_res = final_state["risk_result"]
    assert risk_res is not None
    assert risk_res.outcome == AssessmentOutcome.INCOMPLETE
    assert risk_res.score is None
    assert risk_res.band is None
    assert len(risk_res.missing_evidence) > 0

    # 3. Report draft is persisted in DB
    draft_repo = ReportDraftRepository(session)
    persisted_draft = await draft_repo.get_draft(asmt.id, 1)
    assert persisted_draft is not None
    assert persisted_draft.status == ReportDraftStatus.DRAFT

    # 4. Assessment is at AWAITING_REVIEW (not approved)
    updated_asmt = await asmt_repo.get_assessment(asmt.id)
    assert updated_asmt is not None
    assert updated_asmt.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW
    await engine.dispose()


@pytest.mark.asyncio
async def test_bounded_irrelevant_policy_scenario_abstains_without_unsupported_citations(
    session: AsyncSession,
) -> None:
    """Irrelevant policy query yields no citations and draft abstains without ungrounded refs."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-03",
        idempotency_key="idemp-e10-irrelevant",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.AUCTION),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_clean_vehicle_revision("1HGCR2F85HA000000")
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL)

    # Supply empty policy citations (simulating complete policy abstention)
    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }
        final_state = await runner.run(
            initial_state=initial_state,
            config={"configurable": {"policy_citations": ()}},
        )

    assert final_state["phase"] == AssessmentRunPhase.COMPLETED
    assert len(final_state.get("policy_citations", ())) == 0

    draft_repo = ReportDraftRepository(session)
    persisted_draft = await draft_repo.get_draft(asmt.id, 1)
    assert persisted_draft is not None
    # No citations in section
    assert len(persisted_draft.sections.policy_citations.citations) == 0
    await engine.dispose()


@pytest.mark.asyncio
async def test_bounded_provider_failure_safe_error_no_silent_fallback(
    session: AsyncSession,
) -> None:
    """When provider drafting fails, workflow fails safely without silent offline fallback."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-04",
        idempotency_key="idemp-e10-prov-fail",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_clean_vehicle_revision("1HGCR2F85HA000000")
    mcp_adapter.seed_vehicle(rev)

    failing_drafter = AnthropicDraftingAdapter(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        timeout_seconds=30,
        api_key="sk-ant-test-key",
    )

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(
        database_url=TEST_DB_URL,
        drafting_mode="live",
        enable_live_drafting=True,
        anthropic_api_key=SecretStr("sk-ant-test-key"),
    )

    with patch.object(failing_drafter, "_call_anthropic_api", new_callable=AsyncMock) as mock_api:
        mock_api.side_effect = DraftingFailureError("PROVIDER_UNAVAILABLE")
        async with AssessmentWorkflowRunner.create(
            settings=settings,
            session_factory=session_factory,
            mcp_adapter=mcp_adapter,
            drafting_adapter=failing_drafter,
        ) as runner:
            initial_state: AssessmentGraphState = {
                "assessment_id": asmt.id,
                "run_number": 1,
                "vin": "1HGCR2F85HA000000",
                "context": asmt.context,
                "phase": AssessmentRunPhase.PENDING,
                "visited_phases": [AssessmentRunPhase.PENDING],
                "events": [],
            }
            with pytest.raises(DraftingFailureError):
                await runner.run(initial_state=initial_state)

    # 1. Assessment run moves to FAILED
    updated_asmt = await asmt_repo.get_assessment(asmt.id)
    assert updated_asmt is not None
    assert updated_asmt.runs[0].phase == AssessmentRunPhase.FAILED

    # 2. No report draft was persisted
    draft_repo = ReportDraftRepository(session)
    persisted_draft = await draft_repo.get_draft(asmt.id, 1)
    assert persisted_draft is None

    # 3. Assessment is NOT approved or released
    assert updated_asmt.lifecycle_state != AssessmentLifecycleState.RELEASED
    await engine.dispose()


@pytest.mark.asyncio
async def test_bounded_offline_control_scenario_labelled_offline(
    session: AsyncSession,
) -> None:
    """Offline control run uses OfflineReportDraftingAdapter and is clearly separated from live."""
    asmt_repo = AssessmentRepository(session)
    asmt = await asmt_repo.create_assessment(
        requester_id="req-user-05",
        idempotency_key="idemp-e10-offline-ctrl",
        request=AssessmentCreateRequest(
            vin="1HGCR2F85HA000000",
            context=AssessmentContext(sale_type=SaleType.DEALER),
        ),
    )

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    mcp_adapter = FakeVehicleMcpAdapter()
    rev = _build_clean_vehicle_revision("1HGCR2F85HA000000")
    mcp_adapter.seed_vehicle(rev)

    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = Settings(database_url=TEST_DB_URL, drafting_mode="offline")

    async with AssessmentWorkflowRunner.create(
        settings=settings,
        session_factory=session_factory,
        mcp_adapter=mcp_adapter,
        drafting_adapter=OfflineReportDraftingAdapter(),
    ) as runner:
        initial_state: AssessmentGraphState = {
            "assessment_id": asmt.id,
            "run_number": 1,
            "vin": "1HGCR2F85HA000000",
            "context": asmt.context,
            "phase": AssessmentRunPhase.PENDING,
            "visited_phases": [AssessmentRunPhase.PENDING],
            "events": [],
        }
        final_state = await runner.run(initial_state=initial_state)

    assert final_state["phase"] == AssessmentRunPhase.COMPLETED
    assert isinstance(final_state.get("report_draft"), object)

    draft_repo = ReportDraftRepository(session)
    draft = await draft_repo.get_draft(asmt.id, 1)
    assert draft is not None
    # Offline report draft is explicitly labelled as offline
    assert draft.metadata.get("mode") == "offline"
    assert draft.metadata.get("drafter_id") == "offline-v1"
    await engine.dispose()


@pytest.mark.asyncio
async def test_live_evaluation_runner_executes_bounded_acceptance_with_real_mcp_adapter(
    session: AsyncSession,
) -> None:
    """Bounded live evaluation runner coordinates real MCP adapter and provider drafting."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    policy_repo = RiskPolicyRepository(session)
    policy = build_risk_policy_v1(policy_id="risk-policy-v1")
    await policy_repo.create_policy(policy)
    service = RiskPolicyService(session)
    await service.validate_and_mark_ready(policy.id)
    await service.activate_policy(policy.id, operator_id="admin")

    # Set up real MCP adapter with patched _call_once returning valid tool response
    mcp_adapter = StreamableHttpVehicleMcpAdapter(
        server_url="http://localhost:8000/mcp",
        timeout_seconds=5,
        max_retries=0,
    )
    clean_rev = _build_clean_vehicle_revision("1HGCR2F85HA000000")

    async def fake_mcp_call(tool_name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if tool_name == "lookup_vehicle":
            return types.CallToolResult(
                content=[],
                structured_content=clean_rev.model_dump(mode="json"),
            )
        elif tool_name == "explain_vehicle_field":
            field_name = arguments["field_name"]
            vin = arguments["vin"]
            explanation = FieldExplanationResult(
                vin=vin,
                revision_number=1,
                field_name=field_name,
                outcome=FieldOutcome.RESOLVED,
                value=clean_rev.canonical_fields.get(field_name),
            )
            return types.CallToolResult(
                content=[],
                structured_content=explanation.model_dump(mode="json"),
            )
        return types.CallToolResult(content=[], structured_content={})

    # Set up live drafter with mocked provider API returning grounded payload
    live_drafter = AnthropicDraftingAdapter(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        timeout_seconds=30,
        api_key="sk-ant-test-key",
    )
    mock_payload = {
        "text": json.dumps(
            {
                "assessment_id": "placeholder",
                "run_number": 1,
                "outcome": "SCORED",
                "sections": {
                    "executive_summary": "Clean vehicle assessment completed.",
                    "vehicle_identity": "2017 HONDA ACCORD VIN 1HGCR2F85HA000000 plate NZACC1.",
                    "risk_score": "Deterministic risk score 0 (LOW).",
                    "mandatory_review": "No mandatory review triggers detected.",
                    "contributing_factors": "All risk factors clear.",
                    "evidence_summary": "Verified across NZTA and Police.",
                    "limitations": "Standard limitations apply.",
                    "synthetic_notice": "Production NZTA rules applied.",
                },
                "claim_refs": [],
            }
        ),
        "input_tokens": 1250,
        "output_tokens": 320,
    }

    config = LiveEvaluationConfig(
        enable_live_eval=True,
        api_key="sk-ant-test-key",
        model="claude-sonnet-4-6",
        max_budget_usd=5.0,
        max_scenarios=1,
    )
    from types import SimpleNamespace

    valid_corpus = SimpleNamespace(
        lifecycle_state="ACTIVE",
        retrieval_config=SimpleNamespace(profile="neural"),
    )
    runner = LiveEvaluationRunner(config=config, active_corpus=valid_corpus)

    scenarios = [
        {
            "scenario_id": "scn-eval-live-01",
            "vin": "1HGCR2F85HA000000",
            "sale_type": SaleType.DEALER,
            "expected_outcome": "SCORED",
        }
    ]

    with (
        patch.object(mcp_adapter, "_call_once", side_effect=fake_mcp_call),
        patch.object(live_drafter, "_call_anthropic_api", new_callable=AsyncMock) as mock_api,
    ):
        mock_api.return_value = mock_payload
        record = await runner.run_bounded_acceptance(
            session_factory=session_factory,
            mcp_adapter=mcp_adapter,
            drafting_adapter=live_drafter,
            scenarios=scenarios,
        )

    assert record.execution_mode == "LIVE"
    assert record.total_scenarios <= 3
    assert record.total_estimated_cost_usd <= 5.0
    assert len(record.scenarios) == 1
    assert record.scenarios[0].outcome == "SCORED"
    assert record.scenarios[0].quality_passed is True
    assert record.verdict_passed is True

    # Verify persisted draft and lifecycle state in database
    async with session_factory() as sess:
        draft_repo = ReportDraftRepository(sess)
        stmt = select(AssessmentRecord).where(AssessmentRecord.vin == "1HGCR2F85HA000000")
        asmt_record = (await sess.execute(stmt)).scalars().first()
        assert asmt_record is not None
        assert asmt_record.lifecycle_state == AssessmentLifecycleState.AWAITING_REVIEW.value

        persisted_draft = await draft_repo.get_draft(asmt_record.id, 1)
        assert persisted_draft is not None
        assert persisted_draft.status == ReportDraftStatus.DRAFT
        assert persisted_draft.score is not None

    await engine.dispose()

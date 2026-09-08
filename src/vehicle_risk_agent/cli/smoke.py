"""Comprehensive local smoke verification exercising all domain paths."""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType
from vehicle_risk_agent.cli.seed import seed_database
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.vin import calculate_vin_check_digit
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.persistence.models import (
    AssessmentRecord,
    AssessmentRunRecord,
    ReportDraftRecord,
    ReviewActionRecord,
    RiskResultRecord,
)
from vehicle_risk_agent.persistence.repository import AssessmentRepository
from vehicle_risk_agent.review.models import (
    ApproveReportCommand,
    RejectReportCommand,
    ReportDisposition,
    RequestReinvestigationCommand,
)
from vehicle_risk_agent.review.service import ReviewDecisionService
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand
from vehicle_risk_agent.workflow.runner import AssessmentWorkflowRunner
from vehicle_risk_agent.workflow.state import AssessmentGraphState

logger = logging.getLogger(__name__)


def _make_vin(idx: int) -> str:
    raw = f"7AT0BJ0302000{idx:04d}"
    check = calculate_vin_check_digit(raw)
    assert check is not None
    return raw[:8] + check + raw[9:]


def _make_initial_state(
    assessment_id: str,
    run_number: int,
    vin: str,
    context: AssessmentContext,
) -> AssessmentGraphState:
    return {
        "assessment_id": assessment_id,
        "run_number": run_number,
        "vin": vin,
        "context": context,
        "phase": AssessmentRunPhase.PENDING,
        "visited_phases": [AssessmentRunPhase.PENDING],
        "events": [],
    }


def _make_vehicle_revision(
    vin: str,
    make: str = "HONDA",
    model: str = "ACCORD",
    year: int = 2018,
    ppsr_result: str = "NO_FINANCE_REGISTERED",
    stolen_status: str = "NOT_STOLEN",
    writeoff_status: str = "NOT_WRITTEN_OFF",
) -> VehicleRevisionResponse:
    now = datetime.now(UTC)
    return VehicleRevisionResponse(
        vin=vin,
        revision_id=f"rev-{vin[:8]}",
        revision_number=1,
        material_hash="0" * 64,
        canonical_fields={
            "make": make,
            "model": model,
            "year": year,
            "ppsr_result": ppsr_result,
            "stolen_status": stolen_status,
            "writeoff_status": writeoff_status,
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )


async def run_smoke(database_url: str | None = None) -> dict[str, Any]:
    """Run end-to-end smoke verification across domain paths."""
    settings = Settings()
    db_url = database_url or settings.database_url

    print("==> [1/7] Initializing and seeding local database...")
    seed_result = await seed_database(database_url=db_url, settings=settings)
    print(f"    Seed status: {seed_result['status']}")

    engine = create_async_engine(db_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    fake_mcp = FakeVehicleMcpAdapter()

    # Pre-seed MCP adapter with vehicles for all test scenarios
    clean_vin = _make_vin(1)
    risky_vin = _make_vin(2)
    incomplete_vin = _make_vin(3)
    reinvestigate_vin = _make_vin(4)

    fake_mcp.seed_vehicle(_make_vehicle_revision(clean_vin))
    fake_mcp.seed_vehicle(
        _make_vehicle_revision(
            risky_vin,
            ppsr_result="REGISTERED_SECURITY_INTEREST",
            stolen_status="STOLEN",
            writeoff_status="STATUTORY_WRITEOFF",
        )
    )

    # Incomplete vehicle missing ppsr_result
    now = datetime.now(UTC)
    incomplete_rev = VehicleRevisionResponse(
        vin=incomplete_vin,
        revision_id="rev-incomplete",
        revision_number=1,
        material_hash="1" * 64,
        canonical_fields={
            "make": "NISSAN",
            "model": "LEAF",
            "year": 2020,
        },
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=40,
            band=ConfidenceBand.LOW,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="missing evidence",
        ),
        as_of=now,
        published_at=now,
    )
    fake_mcp.seed_vehicle(incomplete_rev)
    fake_mcp.seed_vehicle(_make_vehicle_revision(reinvestigate_vin))

    test_assessment_ids: list[str] = []

    async with AssessmentWorkflowRunner.create(
        settings, session_factory=session_factory, mcp_adapter=fake_mcp
    ) as runner:
        # --- SCENARIO 1: Clean Vehicle Assessment ---
        print("==> [2/7] Exercising clean vehicle assessment (Low Risk)...")
        run_uid = str(uuid4())[:8]
        async with session_factory() as session:
            repo = AssessmentRepository(session)
            asmt_clean = await repo.create_assessment(
                requester_id="principal-requester-1",
                idempotency_key=f"smoke-clean-{clean_vin}-{run_uid}",
                request=AssessmentCreateRequest(
                    vin=clean_vin,
                    context=AssessmentContext(sale_type=SaleType.DEALER),
                ),
            )
            test_assessment_ids.append(asmt_clean.id)

        clean_state = await runner.run(
            assessment_id=asmt_clean.id,
            run_number=1,
            initial_state=_make_initial_state(
                asmt_clean.id, 1, clean_vin, AssessmentContext(sale_type=SaleType.DEALER)
            ),
        )
        assert clean_state["phase"] == AssessmentRunPhase.COMPLETED
        assert clean_state.get("risk_result") is not None
        assert clean_state["risk_result"].band == RiskBand.LOW
        print(
            f"    Clean vehicle score: {clean_state['risk_result'].score}, "
            f"band: {clean_state['risk_result'].band}"
        )

        # --- SCENARIO 2: Risky Vehicle Assessment ---
        print("==> [3/7] Exercising risky vehicle assessment (High/Critical Risk)...")
        async with session_factory() as session:
            repo = AssessmentRepository(session)
            asmt_risky = await repo.create_assessment(
                requester_id="principal-requester-1",
                idempotency_key=f"smoke-risky-{risky_vin}-{run_uid}",
                request=AssessmentCreateRequest(
                    vin=risky_vin,
                    context=AssessmentContext(sale_type=SaleType.PRIVATE),
                ),
            )
            test_assessment_ids.append(asmt_risky.id)

        risky_state = await runner.run(
            assessment_id=asmt_risky.id,
            run_number=1,
            initial_state=_make_initial_state(
                asmt_risky.id, 1, risky_vin, AssessmentContext(sale_type=SaleType.PRIVATE)
            ),
        )
        assert risky_state["phase"] == AssessmentRunPhase.COMPLETED
        assert risky_state.get("risk_result") is not None
        assert risky_state["risk_result"].band in (RiskBand.HIGH, RiskBand.CRITICAL)
        print(
            f"    Risky vehicle score: {risky_state['risk_result'].score}, "
            f"band: {risky_state['risk_result'].band}"
        )

        # --- SCENARIO 3: Incomplete Evidence Assessment ---
        print("==> [4/7] Exercising incomplete evidence assessment...")
        async with session_factory() as session:
            repo = AssessmentRepository(session)
            asmt_inc = await repo.create_assessment(
                requester_id="principal-requester-1",
                idempotency_key=f"smoke-inc-{incomplete_vin}-{run_uid}",
                request=AssessmentCreateRequest(
                    vin=incomplete_vin,
                    context=AssessmentContext(sale_type=SaleType.AUCTION),
                ),
            )
            test_assessment_ids.append(asmt_inc.id)

        inc_state = await runner.run(
            assessment_id=asmt_inc.id,
            run_number=1,
            initial_state=_make_initial_state(
                asmt_inc.id, 1, incomplete_vin, AssessmentContext(sale_type=SaleType.AUCTION)
            ),
        )
        assert inc_state["phase"] == AssessmentRunPhase.INCOMPLETE
        assert inc_state.get("risk_result") is not None
        assert inc_state["risk_result"].outcome == AssessmentOutcome.INCOMPLETE
        print(f"    Incomplete outcome: {inc_state['risk_result'].outcome}")

        # --- SCENARIO 4: Human Review Decision ---
        print("==> [5/7] Exercising human review approval and rejection...")
        async with session_factory() as session:
            review_svc = ReviewDecisionService(session)
            approve_cmd = ApproveReportCommand(
                assessment_id=asmt_clean.id,
                run_number=1,
                reviewer_id="principal-reviewer-1",
                idempotency_key=f"smoke-review-approve-{run_uid}",
                rationale="Clean history verified against official register data.",
                notes="Approved for customer delivery",
            )
            approve_res = await review_svc.record_review_action(approve_cmd)
            assert approve_res.disposition == ReportDisposition.RELEASED
            print(f"    Clean report disposition: {approve_res.disposition}")

            reject_cmd = RejectReportCommand(
                assessment_id=asmt_risky.id,
                run_number=1,
                reviewer_id="principal-reviewer-1",
                idempotency_key=f"smoke-review-reject-{run_uid}",
                rationale="Vehicle is recorded stolen on register.",
            )
            reject_res = await review_svc.record_review_action(reject_cmd)
            assert reject_res.disposition == ReportDisposition.REJECTED
            print(f"    Risky report disposition: {reject_res.disposition}")

        # --- SCENARIO 5: Reinvestigation Cycle ---
        print("==> [6/7] Exercising reinvestigation flow (Run 2)...")
        async with session_factory() as session:
            repo = AssessmentRepository(session)
            asmt_reinv = await repo.create_assessment(
                requester_id="principal-requester-1",
                idempotency_key=f"smoke-reinv-{reinvestigate_vin}-{run_uid}",
                request=AssessmentCreateRequest(
                    vin=reinvestigate_vin,
                    context=AssessmentContext(sale_type=SaleType.DEALER),
                ),
            )
            test_assessment_ids.append(asmt_reinv.id)

        # Run 1
        await runner.run(
            assessment_id=asmt_reinv.id,
            run_number=1,
            initial_state=_make_initial_state(
                asmt_reinv.id, 1, reinvestigate_vin, AssessmentContext(sale_type=SaleType.DEALER)
            ),
        )

        # Reinvestigate command
        async with session_factory() as session:
            review_svc = ReviewDecisionService(session)
            reinv_cmd = RequestReinvestigationCommand(
                assessment_id=asmt_reinv.id,
                run_number=1,
                reviewer_id="principal-reviewer-1",
                idempotency_key=f"smoke-reinvestigate-{run_uid}",
                rationale="Need additional evidence check for recent odometer read",
                evidence_targets=("odometer_reading",),
            )
            reinv_res = await review_svc.record_review_action(reinv_cmd)
            assert reinv_res.next_run_number == 2
            print(f"    Reinvestigation triggered for run: {reinv_res.next_run_number}")

        # Run 2
        run2_state = await runner.run(
            assessment_id=asmt_reinv.id,
            run_number=2,
            initial_state=_make_initial_state(
                asmt_reinv.id, 2, reinvestigate_vin, AssessmentContext(sale_type=SaleType.DEALER)
            ),
        )
        assert run2_state["run_number"] == 2
        print(f"    Run 2 completed with phase: {run2_state['phase']}")

    # --- SCENARIO 6: Restart & Idempotency ---
    print("==> [7/7] Verifying restart idempotency and tearing down smoke records...")
    re_seed = await seed_database(database_url=db_url, settings=settings)
    assert re_seed["status"] in ("seeded", "ok")

    # Teardown smoke assessments
    async with session_factory() as session:
        for aid in test_assessment_ids:
            await session.execute(
                delete(ReviewActionRecord).where(ReviewActionRecord.assessment_id == aid)
            )
            await session.execute(
                delete(ReportDraftRecord).where(ReportDraftRecord.assessment_id == aid)
            )
            await session.execute(
                delete(RiskResultRecord).where(RiskResultRecord.assessment_id == aid)
            )
            await session.execute(
                delete(AssessmentRunRecord).where(AssessmentRunRecord.assessment_id == aid)
            )
            await session.execute(delete(AssessmentRecord).where(AssessmentRecord.id == aid))
        await session.commit()

    await engine.dispose()
    print("    Teardown completed cleanly.")
    return {
        "status": "success",
        "exercised": [
            "clean",
            "risky",
            "incomplete",
            "review",
            "reinvestigate",
            "restart",
            "teardown",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local smoke verification")
    parser.add_argument("--database-url", type=str, default=None)
    args = parser.parse_args()

    result = asyncio.run(run_smoke(database_url=args.database_url))
    print(f"Smoke verification result: {result['status']}")
    sys.exit(0)


if __name__ == "__main__":
    main()

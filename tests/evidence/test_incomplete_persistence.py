"""Tests for persisting ordered missing findings and enforcing score prohibition on INCOMPLETE."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import (
    VehicleEvidenceRepository,
    create_evidence_snapshot,
)
from vehicle_risk_agent.evidence.sufficiency import (
    IncompleteAssessmentReport,
    MissingEvidenceFinding,
    MissingEvidenceReason,
    SufficiencyOutcome,
    evaluate_evidence_sufficiency,
)
from vehicle_risk_agent.persistence.models import AssessmentRecord, Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def test_incomplete_assessment_report_prohibits_score_and_band() -> None:
    """IncompleteAssessmentReport strictly enforces that no score or band exists on INCOMPLETE."""
    finding = MissingEvidenceFinding(
        field_name="stolen_status",
        reason=MissingEvidenceReason.UNKNOWN,
        details="Stolen status is reported as UNKNOWN in source records.",
    )
    report = IncompleteAssessmentReport(
        assessment_id="asmt-inc-01",
        run_number=1,
        vin="7AT0BK00X00000001",
        outcome=SufficiencyOutcome.INCOMPLETE,
        missing_findings=(finding,),
    )

    assert report.outcome == SufficiencyOutcome.INCOMPLETE
    assert report.missing_findings[0].field_name == "stolen_status"

    # Strictness check: Attempting to pass risk_score or risk_band must raise ValidationError
    with pytest.raises(ValidationError):
        IncompleteAssessmentReport(
            assessment_id="asmt-inc-01",
            run_number=1,
            vin="7AT0BK00X00000001",
            outcome=SufficiencyOutcome.INCOMPLETE,
            missing_findings=(finding,),
            risk_score=50,  # type: ignore[call-arg]
        )


@pytest.mark.asyncio
async def test_persist_and_retrieve_incomplete_evidence_findings(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Persist ordered missing evidence findings with snapshot and retrieve accurately."""
    now = datetime.now(UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-inc-1",
        revision_number=1,
        material_hash="f" * 64,
        canonical_fields={
            "make": "SUBARU",
            "model": "LEGACY",
            "year": 2016,
            "ppsr_result": "UNKNOWN",
            "stolen_status": "UNKNOWN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={},
        conflicts=[],
        confidence=ConfidenceAssessment(
            score=70,
            band=ConfidenceBand.MEDIUM,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="partial",
        ),
        as_of=now,
        published_at=now,
    )

    snapshot = create_evidence_snapshot("asmt-inc-01", 1, rev)
    sufficiency = evaluate_evidence_sufficiency(snapshot)

    assert sufficiency.outcome == SufficiencyOutcome.INCOMPLETE
    # Findings must be sorted deterministically by field_name: ppsr_result, then stolen_status
    assert len(sufficiency.missing_findings) == 2
    assert sufficiency.missing_findings[0].field_name == "ppsr_result"
    assert sufficiency.missing_findings[1].field_name == "stolen_status"

    async with session_factory() as session:
        asmt = AssessmentRecord(
            id="asmt-inc-01",
            requester_id="req-1",
            vin="7AT0BK00X00000001",
            context_json="{}",
        )
        session.add(asmt)
        await session.flush()

        repo = VehicleEvidenceRepository(session)
        await repo.save_snapshot(snapshot)
        await repo.save_sufficiency_result("asmt-inc-01", 1, sufficiency)

        retrieved_suff = await repo.get_sufficiency_result("asmt-inc-01", 1)
        assert retrieved_suff is not None
        assert retrieved_suff.outcome == SufficiencyOutcome.INCOMPLETE
        assert len(retrieved_suff.missing_findings) == 2
        assert retrieved_suff.missing_findings[0].field_name == "ppsr_result"
        assert retrieved_suff.missing_findings[1].field_name == "stolen_status"

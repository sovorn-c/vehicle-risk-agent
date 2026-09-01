"""Tests for persisting and retrieving immutable Vehicle Evidence Snapshots."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    ProvenanceLink,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import (
    VehicleEvidenceRepository,
    VehicleEvidenceSnapshot,
    create_evidence_snapshot,
)
from vehicle_risk_agent.persistence.models import Base

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


@pytest.fixture
def sample_revision() -> VehicleRevisionResponse:
    now = datetime.now(UTC)
    provenance = ProvenanceLink(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        retrieved_at=now,
        synthetic=True,
    )
    confidence = ConfidenceAssessment(
        score=92,
        band=ConfidenceBand.HIGH,
        field_scores={"make": 95, "model": 95, "year": 90},
        field_components={
            "make": {"authority": 40, "freshness": 25, "agreement": 20, "validation": 10}
        },
        rule_version="conf-v1",
        explanation="Validated NZ register record.",
    )
    return VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-001",
        revision_number=1,
        material_hash="e" * 64,
        canonical_fields={
            "make": "TOYOTA",
            "model": "AQUA",
            "year": 2019,
            "ppsr_result": "NO_FINANCE_REGISTERED",
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        },
        field_provenance={"make": [provenance]},
        conflicts=[],
        confidence=confidence,
        as_of=now,
        published_at=now,
        synthetic_notice="SYNTHETIC PROVENANCE NOTICE",
    )


@pytest.mark.asyncio
async def test_create_and_persist_evidence_snapshot(
    session_factory: async_sessionmaker[AsyncSession],
    sample_revision: VehicleRevisionResponse,
) -> None:
    """Evidence snapshot captures all upstream revision properties and persists immutably."""
    snapshot = create_evidence_snapshot(
        assessment_id="asmt-001",
        run_number=1,
        revision=sample_revision,
    )

    assert isinstance(snapshot, VehicleEvidenceSnapshot)
    assert snapshot.assessment_id == "asmt-001"
    assert snapshot.run_number == 1
    assert snapshot.vin == "7AT0BK00X00000001"
    assert snapshot.canonical_fields["make"] == "TOYOTA"
    assert snapshot.synthetic_notice == "SYNTHETIC PROVENANCE NOTICE"

    async with session_factory() as session:
        repo = VehicleEvidenceRepository(session)
        await repo.save_snapshot(snapshot)

        retrieved = await repo.get_snapshot("asmt-001", 1)
        assert retrieved is not None
        assert retrieved.vin == "7AT0BK00X00000001"
        assert retrieved.canonical_fields["model"] == "AQUA"
        assert retrieved.confidence.score == 92
        assert retrieved.synthetic_notice == "SYNTHETIC PROVENANCE NOTICE"
        assert len(retrieved.field_provenance["make"]) == 1
        assert retrieved.field_provenance["make"][0].observation_id == "obs-nzta-001"

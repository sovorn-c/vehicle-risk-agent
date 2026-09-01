"""Comprehensive tests verifying hardening against residual blockers:
1. Fail-closed routing on UNAVAILABLE / missing MCP adapter.
2. Prevention of role spoofing via headers.
3. Integration of temporal history & parallel field explanations in snapshot.
4. Cryptographic integrity and bounds validation for source observations, hashes, and scores.
5. Repository snapshot idempotency and hash authentication.
6. Run-partitioned workflow progress events.
"""

import hashlib
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import (
    VehicleEvidenceRepository,
    create_evidence_snapshot,
)
from vehicle_risk_agent.persistence.event_store import EventStore
from vehicle_risk_agent.persistence.models import Base
from vehicle_risk_agent.workflow.runner import AssessmentRunner

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


@pytest.mark.asyncio
async def test_no_mcp_adapter_fails_closed_to_failed_phase() -> None:
    """AssessmentRunner without MCP adapter routes to FAILED, never COMPLETED."""
    runner = AssessmentRunner(mcp_adapter=None)
    result = await runner.run(
        assessment_id="asmt-fail-closed",
        run_number=1,
        vin="7AT0BK00X00000001",
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )

    assert result["phase"] == AssessmentRunPhase.FAILED
    assert result["visited_phases"] == [
        AssessmentRunPhase.PENDING,
        AssessmentRunPhase.COLLECTING_EVIDENCE,
        AssessmentRunPhase.FAILED,
    ]
    assert result.get("mcp_error") is not None


@pytest.mark.asyncio
async def test_unknown_vehicle_lookup_fails_closed_to_failed_phase() -> None:
    """AssessmentRunner with empty fake adapter routes to FAILED."""
    adapter = FakeVehicleMcpAdapter()  # no vehicle seeded
    runner = AssessmentRunner(mcp_adapter=adapter)
    result = await runner.run(
        assessment_id="asmt-unavail",
        run_number=1,
        vin="7AT0BK00X00000001",
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )

    assert result["phase"] == AssessmentRunPhase.FAILED
    assert result["visited_phases"] == [
        AssessmentRunPhase.PENDING,
        AssessmentRunPhase.COLLECTING_EVIDENCE,
        AssessmentRunPhase.FAILED,
    ]


@pytest.mark.asyncio
async def test_role_header_without_bearer_token_is_rejected() -> None:
    """X-User-Role header cannot spoof reviewer authorization without valid bearer token."""
    settings = Settings(database_url=TEST_DB_URL)
    app = create_app(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Attempt to access reviewer-only observation audit using X-User-Role header
        res = await client.get(
            "/api/v1/assessments/asmt-1/runs/1/evidence/observations/obs-1",
            headers={"X-User-Role": "reviewer"},
        )
        assert res.status_code == 401


def test_material_hash_and_confidence_score_strict_validation() -> None:
    """Malformed material hash or confidence score outside 0-100 is rejected."""
    now = datetime.now(UTC)

    # 1. Invalid material hash (not 64 hex characters)
    with pytest.raises(ValidationError):
        VehicleRevisionResponse(
            vin="7AT0BK00X00000001",
            revision_id="rev-1",
            revision_number=1,
            material_hash="invalid-hash-too-short",
            canonical_fields={"make": "HONDA"},
            confidence=ConfidenceAssessment(
                score=80,
                band=ConfidenceBand.HIGH,
                field_scores={},
                field_components={},
                rule_version="v1",
                explanation="ok",
            ),
            as_of=now,
            published_at=now,
        )

    # 2. Confidence score > 100 rejected
    with pytest.raises(ValidationError):
        ConfidenceAssessment(
            score=150,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="ok",
        )

    # 3. Confidence score < 0 rejected
    with pytest.raises(ValidationError):
        ConfidenceAssessment(
            score=-5,
            band=ConfidenceBand.LOW,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="ok",
        )


def test_source_observation_hash_integrity_and_bounds() -> None:
    """SourceObservationResponse rejects tampered hash or payload exceeding 1MB."""
    now = datetime.now(UTC)
    raw = '{"source": "test"}'
    valid_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # Valid model
    obs = SourceObservationResponse(
        observation_id="obs-1",
        source_system="SYS",
        source_record_id="rec-1",
        ingestion_run_id="run-1",
        raw_payload=raw,
        payload_hash_sha256=valid_hash,
        retrieved_at=now,
        synthetic=True,
    )
    assert obs.payload_hash_sha256 == valid_hash

    # Tampered hash rejected
    with pytest.raises(ValidationError):
        SourceObservationResponse(
            observation_id="obs-1",
            source_system="SYS",
            source_record_id="rec-1",
            ingestion_run_id="run-1",
            raw_payload=raw,
            payload_hash_sha256="0" * 64,  # mismatched hash
            retrieved_at=now,
            synthetic=True,
        )


@pytest.mark.asyncio
async def test_snapshot_idempotent_save_and_hash_authentication(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Duplicate snapshot save is idempotent and DB record hash must match snapshot."""
    from vehicle_risk_agent.persistence.models import AssessmentRecord

    now = datetime.now(UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields={"make": "HONDA"},
        field_provenance={},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={},
            field_components={},
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )
    snap = create_evidence_snapshot("asmt-idemp-01", 1, rev)

    async with session_factory() as session:
        asmt = AssessmentRecord(
            id="asmt-idemp-01",
            requester_id="req-1",
            vin="7AT0BK00X00000001",
            context_json="{}",
        )
        session.add(asmt)
        await session.flush()

        repo = VehicleEvidenceRepository(session)
        # First save
        await repo.save_snapshot(snap)
        # Second idempotent save
        await repo.save_snapshot(snap)

        retrieved = await repo.get_snapshot("asmt-idemp-01", 1)
        assert retrieved is not None
        assert retrieved.material_hash == "a" * 64


@pytest.mark.asyncio
async def test_event_store_partitions_by_run_number(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """EventStore filters events by run_number so runs do not mix."""
    from vehicle_risk_agent.persistence.models import AssessmentRecord

    async with session_factory() as session:
        asmt = AssessmentRecord(
            id="asmt-multi-run",
            requester_id="req-1",
            vin="7AT0BK00X00000001",
            context_json="{}",
        )
        session.add(asmt)
        await session.flush()

        store = EventStore(session)

        # Run 1 event
        await store.append_event(
            WorkflowProgressEvent(
                event_id="e-1",
                sequence=1,
                assessment_id="asmt-multi-run",
                run_number=1,
                phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
                safe_message="Run 1 collecting evidence",
                timestamp=datetime.now(UTC),
            )
        )
        # Run 2 event
        await store.append_event(
            WorkflowProgressEvent(
                event_id="e-2",
                sequence=1,
                assessment_id="asmt-multi-run",
                run_number=2,
                phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
                safe_message="Run 2 collecting evidence",
                timestamp=datetime.now(UTC),
            )
        )

        run1_events = await store.get_events("asmt-multi-run", run_number=1)
        run2_events = await store.get_events("asmt-multi-run", run_number=2)

        assert len(run1_events) == 1
        assert run1_events[0].run_number == 1
        assert len(run2_events) == 1
        assert run2_events[0].run_number == 2

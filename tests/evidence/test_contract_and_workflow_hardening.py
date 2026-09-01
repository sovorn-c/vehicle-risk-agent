"""Comprehensive tests verifying hardening against residual blockers:
1. Fail-closed routing on UNAVAILABLE / missing MCP adapter.
2. Prevention of role spoofing via headers.
3. Integration of temporal history & parallel field explanations in snapshot.
4. Cryptographic integrity and bounds validation for source observations, hashes, and scores.
5. Repository snapshot idempotency and hash authentication.
6. Run-partitioned workflow progress events.
"""

import hashlib
import hmac
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import (
    FakeVehicleMcpAdapter,
    McpAdapterError,
    StreamableHttpVehicleMcpAdapter,
    UnavailableVehicleMcpAdapter,
)
from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.models import AssessmentContext, SaleType
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.domain.assessment import AssessmentRunPhase
from vehicle_risk_agent.domain.events import WorkflowProgressEvent
from vehicle_risk_agent.evidence.audit import (
    ProvenanceMismatchError,
    SourceObservationAuditService,
)
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    FieldExplanationResult,
    ProvenanceLink,
    SafeErrorCategory,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import (
    SnapshotIntegrityError,
    VehicleEvidenceRepository,
    create_evidence_snapshot,
)
from vehicle_risk_agent.evidence.sufficiency import (
    SufficiencyOutcome,
    evaluate_evidence_sufficiency,
)
from vehicle_risk_agent.persistence.event_store import EventStore
from vehicle_risk_agent.persistence.models import Base, VehicleEvidenceSnapshotRecord
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

    # UTF-8 byte size, not only character count, is bounded
    oversized = "é" * 524_289
    with pytest.raises(ValidationError):
        SourceObservationResponse(
            observation_id="obs-1",
            source_system="SYS",
            source_record_id="rec-1",
            ingestion_run_id="run-1",
            raw_payload=oversized,
            payload_hash_sha256=hashlib.sha256(oversized.encode("utf-8")).hexdigest(),
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

        # A replayed node may regenerate the same sequence after a crash.
        await store.append_event(
            WorkflowProgressEvent(
                event_id="e-1-retry",
                sequence=1,
                assessment_id="asmt-multi-run",
                run_number=1,
                phase=AssessmentRunPhase.COLLECTING_EVIDENCE,
                safe_message="duplicate must be ignored",
                timestamp=datetime.now(UTC),
            )
        )

        run1_events = await store.get_events("asmt-multi-run", run_number=1)
        run2_events = await store.get_events("asmt-multi-run", run_number=2)

        assert len(run1_events) == 1
        assert run1_events[0].run_number == 1
        assert len(run2_events) == 1
        assert run2_events[0].run_number == 2


def _revision_for_hardening(
    *,
    revision_id: str = "rev-hardening-1",
    make: str = "HONDA",
    canonical_fields: dict[str, Any] | None = None,
    field_provenance: dict[str, tuple[ProvenanceLink, ...] | list[ProvenanceLink]] | None = None,
) -> VehicleRevisionResponse:
    now = datetime.now(UTC)
    return VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id=revision_id,
        revision_number=1,
        material_hash=("a" if revision_id.endswith("1") else "b") * 64,
        canonical_fields=canonical_fields or {"make": make},
        field_provenance=field_provenance or {},
        conflicts=(),
        confidence=ConfidenceAssessment(
            score=90,
            band=ConfidenceBand.HIGH,
            field_scores={"make": 90},
            field_components={},
            rule_version="v1",
            explanation="verified",
        ),
        as_of=now,
        published_at=now,
    )


def test_snapshot_nested_values_are_immutable() -> None:
    """Snapshot maps and nested maps cannot be changed after construction."""
    snapshot = create_evidence_snapshot(
        "asmt-immutable",
        1,
        _revision_for_hardening(
            canonical_fields={"nested": {"value": "before"}},
        ),
    )

    with pytest.raises(TypeError):
        snapshot.canonical_fields["nested"]["value"] = "after"
    with pytest.raises(TypeError):
        snapshot.confidence.field_scores["make"] = 1


def test_sufficiency_rejects_unverified_required_values() -> None:
    """Blank, unknown, unresolved, and non-string required values cannot pass sufficiency."""
    for value in ("", "unknown", "UNRESOLVED", object()):
        fields: dict[str, object] = {
            "ppsr_result": value,
            "stolen_status": "NOT_STOLEN",
            "writeoff_status": "NOT_WRITTEN_OFF",
        }
        result = evaluate_evidence_sufficiency(
            create_evidence_snapshot(
                "asmt-sufficiency", 1, _revision_for_hardening(canonical_fields=fields)
            )
        )
        assert result.outcome == SufficiencyOutcome.INCOMPLETE
        assert not result.is_sufficient


class FailingExplanationAdapter(FakeVehicleMcpAdapter):
    """Fake adapter that exposes a field explanation failure."""

    async def explain_vehicle_field(self, _vin: str, _field_name: str) -> FieldExplanationResult:
        raise McpAdapterError(
            category=SafeErrorCategory.PIPELINE_UNAVAILABLE,
            message="raw upstream credentials must not escape",
            retryable=True,
            remediation="raw remediation must not escape",
        )


@pytest.mark.asyncio
async def test_field_explanation_failure_fails_run_instead_of_completing() -> None:
    """A required MCP explanation failure must fail closed, not create an empty explanation set."""
    adapter = FailingExplanationAdapter()
    adapter.seed_vehicle(
        _revision_for_hardening(
            canonical_fields={
                "ppsr_result": "OK",
                "stolen_status": "NOT_STOLEN",
                "writeoff_status": "NOT_WRITTEN_OFF",
            }
        )
    )
    result = await AssessmentRunner(mcp_adapter=adapter).run(
        assessment_id="asmt-explanation-failure",
        run_number=1,
        vin="7AT0BK00X00000001",
        context=AssessmentContext(sale_type=SaleType.DEALER),
    )
    assert result["phase"] == AssessmentRunPhase.FAILED
    assert result["mcp_error"].message != "raw upstream credentials must not escape"
    assert "raw upstream" not in " ".join(event.safe_message for event in result["events"])


def test_unconfigured_mcp_adapter_is_unavailable_not_fake() -> None:
    """Application wiring must fail closed when no MCP endpoint is configured."""
    app = create_app(settings=Settings())
    assert isinstance(app.state.mcp_adapter, UnavailableVehicleMcpAdapter)


def test_configured_mcp_endpoint_uses_real_adapter() -> None:
    """A configured endpoint must wire the official MCP client adapter."""
    app = create_app(settings=Settings(mcp_server_url="http://mcp:8000/mcp"))
    assert isinstance(app.state.mcp_adapter, StreamableHttpVehicleMcpAdapter)


def test_mcp_settings_reject_non_positive_values() -> None:
    """Timeout and retry settings cannot disable defensive bounds."""
    with pytest.raises(ValidationError):
        Settings(mcp_timeout_seconds=0)
    with pytest.raises(ValidationError):
        Settings(mcp_max_retries=-1)
    with pytest.raises(ValidationError):
        Settings(mcp_initial_backoff=0)


@pytest.mark.asyncio
async def test_same_run_snapshot_cannot_be_replaced(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Saving a different snapshot for an existing run must preserve the first snapshot."""
    from vehicle_risk_agent.persistence.models import AssessmentRecord

    async with session_factory() as session:
        session.add(
            AssessmentRecord(
                id="asmt-immutable-save",
                requester_id="req-1",
                vin="7AT0BK00X00000001",
                context_json="{}",
            )
        )
        await session.flush()
        repo = VehicleEvidenceRepository(session)
        first = create_evidence_snapshot("asmt-immutable-save", 1, _revision_for_hardening())
        second = create_evidence_snapshot(
            "asmt-immutable-save",
            1,
            _revision_for_hardening(revision_id="rev-hardening-2", make="TOYOTA"),
        )
        await repo.save_snapshot(first)
        with pytest.raises(ValueError, match="immutable"):
            await repo.save_snapshot(second)
        stored = await repo.get_snapshot("asmt-immutable-save", 1)
        assert stored is not None
        assert stored.revision_id == first.revision_id


@pytest.mark.asyncio
async def test_snapshot_json_tampering_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Changing persisted snapshot JSON must fail local integrity verification."""
    import json

    from vehicle_risk_agent.persistence.models import AssessmentRecord

    async with session_factory() as session:
        session.add(
            AssessmentRecord(
                id="asmt-tamper",
                requester_id="req-1",
                vin="7AT0BK00X00000001",
                context_json="{}",
            )
        )
        await session.flush()
        repo = VehicleEvidenceRepository(session)
        snapshot = create_evidence_snapshot("asmt-tamper", 1, _revision_for_hardening())
        await repo.save_snapshot(snapshot)
        record = (
            await session.execute(
                sa.select(VehicleEvidenceSnapshotRecord).where(
                    VehicleEvidenceSnapshotRecord.assessment_id == "asmt-tamper"
                )
            )
        ).scalar_one()
        data = json.loads(record.snapshot_data_json)
        data["vin"] = "1HGCM82633A004352"
        tampered_json = json.dumps(data)
        record.snapshot_data_json = tampered_json
        record.snapshot_integrity_hash = hmac.new(
            b"an-attacker-does-not-have-the-server-key",
            tampered_json.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        await session.commit()
        with pytest.raises(SnapshotIntegrityError):
            await repo.get_snapshot("asmt-tamper", 1)


@pytest.mark.asyncio
async def test_provenance_metadata_tampering_is_rejected() -> None:
    """A same-ID observation with different source metadata must not pass audit."""
    now = datetime.now(UTC)
    link = ProvenanceLink(
        observation_id="obs-1",
        source_system="NZTA_MVR",
        source_record_id="record-1",
        retrieved_at=now,
        synthetic=True,
    )
    snapshot = create_evidence_snapshot(
        "asmt-provenance",
        1,
        _revision_for_hardening(field_provenance={"make": [link]}),
    )
    raw = "{}"

    class WrongMetadataAdapter(FakeVehicleMcpAdapter):
        async def get_source_observation(self, observation_id: str) -> SourceObservationResponse:
            return SourceObservationResponse(
                observation_id=observation_id,
                source_system="EVIL",
                source_record_id="record-1",
                ingestion_run_id="run-1",
                raw_payload=raw,
                payload_hash_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                retrieved_at=now,
                synthetic=True,
            )

    with pytest.raises(ProvenanceMismatchError):
        await SourceObservationAuditService().resolve_linked_observation(
            snapshot, "obs-1", WrongMetadataAdapter()
        )

    class WrongIdAdapter(FakeVehicleMcpAdapter):
        async def get_source_observation(self, _observation_id: str) -> SourceObservationResponse:
            return SourceObservationResponse(
                observation_id="another-observation",
                source_system="NZTA_MVR",
                source_record_id="record-1",
                ingestion_run_id="run-1",
                raw_payload=raw,
                payload_hash_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                retrieved_at=now,
                synthetic=True,
            )

    with pytest.raises(ProvenanceMismatchError):
        await SourceObservationAuditService().resolve_linked_observation(
            snapshot, "obs-1", WrongIdAdapter()
        )


def test_snapshot_record_contains_integrity_hash() -> None:
    """The persistence model stores a local hash of the serialized snapshot."""
    assert "snapshot_integrity_hash" in VehicleEvidenceSnapshotRecord.__table__.columns

"""API integration tests for reviewer-authorized exact source observation inspection."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import sqlalchemy as sa
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vehicle_risk_agent.adapters.mcp import FakeVehicleMcpAdapter
from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.deps import get_db_session, get_mcp_adapter
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evidence.models import (
    ConfidenceAssessment,
    ConfidenceBand,
    ProvenanceLink,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.snapshot import (
    VehicleEvidenceRepository,
    create_evidence_snapshot,
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


@pytest.fixture
def fake_mcp_adapter() -> FakeVehicleMcpAdapter:
    adapter = FakeVehicleMcpAdapter()
    now = datetime.now(UTC)
    obs = SourceObservationResponse(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        ingestion_run_id="ingest-001",
        raw_payload='{"make": "HONDA", "model": "FIT"}',
        payload_hash_sha256="c" * 64,
        retrieved_at=now,
        synthetic=True,
    )
    adapter.seed_source_observation(obs)
    return adapter


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker[AsyncSession],
    fake_mcp_adapter: FakeVehicleMcpAdapter,
) -> AsyncGenerator[AsyncClient, None]:
    settings = Settings(database_url=TEST_DB_URL, secret_key="test-secret")
    app = create_app(settings=settings)

    async def override_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    def override_mcp_adapter() -> FakeVehicleMcpAdapter:
        return fake_mcp_adapter

    app.dependency_overrides[get_db_session] = override_db_session
    app.dependency_overrides[get_mcp_adapter] = override_mcp_adapter

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_reviewer_can_inspect_linked_source_observation(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Authorized reviewer can inspect exact source observation linked to assessment evidence."""
    now = datetime.now(UTC)
    p = ProvenanceLink(
        observation_id="obs-nzta-001",
        source_system="NZTA_MVR",
        source_record_id="rec-001",
        retrieved_at=now,
        synthetic=True,
    )
    rev = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="d" * 64,
        canonical_fields={"make": "HONDA", "model": "FIT"},
        field_provenance={"make": [p]},
        conflicts=[],
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

    async with session_factory() as session:
        asmt = AssessmentRecord(
            id="asmt-audit-01",
            requester_id="req-1",
            vin="7AT0BK00X00000001",
            context_json="{}",
        )
        session.add(asmt)
        await session.flush()

        repo = VehicleEvidenceRepository(session)
        snap = create_evidence_snapshot("asmt-audit-01", 1, rev)
        await repo.save_snapshot(snap)

    headers = {"X-User-Role": "reviewer", "X-User-Id": "reviewer-1"}
    response = await client.get(
        "/api/v1/assessments/asmt-audit-01/runs/1/evidence/observations/obs-nzta-001",
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["observation_id"] == "obs-nzta-001"
    assert data["source_system"] == "NZTA_MVR"
    assert data["synthetic"] is True
    assert "HONDA" in data["raw_payload"]
    assert data["payload_hash_sha256"] == "c" * 64


@pytest.mark.asyncio
async def test_reviewer_inspecting_unlinked_observation_returns_404(
    client: AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Attempting to inspect an unlinked observation returns 404 Not Found."""
    now = datetime.now(UTC)
    rev = VehicleRevisionResponse(
        vin="7AT0BK00X00000001",
        revision_id="rev-1",
        revision_number=1,
        material_hash="d" * 64,
        canonical_fields={"make": "HONDA"},
        field_provenance={},
        conflicts=[],
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

    async with session_factory() as session:
        asmt = AssessmentRecord(
            id="asmt-audit-02",
            requester_id="req-1",
            vin="7AT0BK00X00000001",
            context_json="{}",
        )
        session.add(asmt)
        await session.flush()

        repo = VehicleEvidenceRepository(session)
        snap = create_evidence_snapshot("asmt-audit-02", 1, rev)
        await repo.save_snapshot(snap)

    headers = {"X-User-Role": "reviewer", "X-User-Id": "reviewer-1"}
    response = await client.get(
        "/api/v1/assessments/asmt-audit-02/runs/1/evidence/observations/obs-unlinked-999",
        headers=headers,
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_non_reviewer_role_forbidden_from_observation_audit(
    client: AsyncClient,
) -> None:
    """Non-reviewer role (e.g. requester) is forbidden from raw observation inspection."""
    headers = {"X-User-Role": "requester", "X-User-Id": "req-1"}
    response = await client.get(
        "/api/v1/assessments/asmt-audit-01/runs/1/evidence/observations/obs-nzta-001",
        headers=headers,
    )
    assert response.status_code == 403

"""Tests for Policy Corpus creation, validation, activation, and inspection API routes."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.persistence.models import Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def app_client() -> AsyncIterator[AsyncClient]:
    """Provide an AsyncClient for FastAPI application with test database."""
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    settings = Settings(database_url=TEST_DB_URL)
    app = create_app(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await engine.dispose()


@pytest.mark.asyncio
async def test_maintainer_creates_validates_and_activates_corpus(app_client: AsyncClient) -> None:
    """Policy Corpus Maintainer creates DRAFT, validates to READY, and activates corpus."""
    # 1. Register source and ingest snapshot
    await app_client.post(
        "/api/v1/policy/sources",
        json={
            "id": "nz-fta-1986",
            "title": "Fair Trading Act 1986",
            "issuing_authority": "Parliament of New Zealand",
            "jurisdiction": "NZ",
            "canonical_origin": "https://example.com/fta",
            "authority_classification": "PRIMARY_LEGISLATION",
            "reuse_terms": "CC BY 4.0",
            "expected_update_cadence": "ADHOC",
        },
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    snap_res = await app_client.post(
        "/api/v1/policy/sources/nz-fta-1986/snapshots",
        json={"raw_content": "# FTA 1986\n## Section 9: Misleading conduct\nProhibited in trade."},
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    snap_id = snap_res.json()["id"]

    # 2. Create corpus (DRAFT)
    res_create = await app_client.post(
        "/api/v1/policy/corpora",
        json={
            "id": "corpus-2026-v1",
            "name": "NZ 2026 Baseline Corpus",
            "description": "Initial policy baseline",
            "snapshot_ids": [snap_id],
        },
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res_create.status_code == 201, res_create.text
    assert res_create.json()["lifecycle_state"] == "DRAFT"

    # 3. Mark READY
    res_ready = await app_client.post(
        "/api/v1/policy/corpora/corpus-2026-v1/ready",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res_ready.status_code == 200, res_ready.text
    assert res_ready.json()["lifecycle_state"] == "READY"

    # 4. Activate
    res_act = await app_client.post(
        "/api/v1/policy/corpora/corpus-2026-v1/activate",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res_act.status_code == 200, res_act.text
    data = res_act.json()
    assert data["active_corpus"]["id"] == "corpus-2026-v1"
    assert data["active_corpus"]["lifecycle_state"] == "ACTIVE"
    assert data["retired_corpus"] is None

    # 5. Check active corpus endpoint
    res_get_active = await app_client.get(
        "/api/v1/policy/corpora/active",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert res_get_active.status_code == 200
    assert res_get_active.json()["id"] == "corpus-2026-v1"


@pytest.mark.asyncio
async def test_non_maintainer_cannot_activate_corpus(app_client: AsyncClient) -> None:
    """Requesters and Reviewers cannot mutate corpus state."""
    res = await app_client.post(
        "/api/v1/policy/corpora/corpus-2026-v1/activate",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_retired_corpus_inspectable_for_replay(app_client: AsyncClient) -> None:
    """When a new corpus activates, the retired corpus remains inspectable by ID."""
    # Register source & snapshot
    await app_client.post(
        "/api/v1/policy/sources",
        json={
            "id": "nz-fta-1986",
            "title": "Fair Trading Act 1986",
            "issuing_authority": "Parliament of New Zealand",
            "jurisdiction": "NZ",
            "canonical_origin": "https://example.com/fta",
            "authority_classification": "PRIMARY_LEGISLATION",
            "reuse_terms": "CC BY 4.0",
            "expected_update_cadence": "ADHOC",
        },
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    snap_res = await app_client.post(
        "/api/v1/policy/sources/nz-fta-1986/snapshots",
        json={"raw_content": "# FTA 1986\n## Section 9: Misleading conduct\nProhibited in trade."},
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    snap_id = snap_res.json()["id"]

    # Setup corpus 1 -> activate
    await app_client.post(
        "/api/v1/policy/corpora",
        json={"id": "c1", "name": "C1", "description": "D", "snapshot_ids": [snap_id]},
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    await app_client.post(
        "/api/v1/policy/corpora/c1/ready",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    await app_client.post(
        "/api/v1/policy/corpora/c1/activate",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )

    # Setup corpus 2 -> activate
    await app_client.post(
        "/api/v1/policy/corpora",
        json={"id": "c2", "name": "C2", "description": "D", "snapshot_ids": [snap_id]},
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    await app_client.post(
        "/api/v1/policy/corpora/c2/ready",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    res_act2 = await app_client.post(
        "/api/v1/policy/corpora/c2/activate",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res_act2.json()["retired_corpus"]["id"] == "c1"

    # Inspect retired corpus 1
    res_c1 = await app_client.get(
        "/api/v1/policy/corpora/c1",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert res_c1.status_code == 200
    assert res_c1.json()["lifecycle_state"] == "RETIRED"

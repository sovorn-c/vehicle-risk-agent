"""Tests for policy sources and snapshots API routes, authorization, and idempotency."""

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
async def test_maintainer_creates_and_retrieves_policy_source(app_client: AsyncClient) -> None:
    """A policy corpus maintainer can register a new policy source and retrieve it."""
    payload = {
        "id": "nz-legislation-fta-1986",
        "title": "Fair Trading Act 1986",
        "issuing_authority": "Parliament of New Zealand",
        "jurisdiction": "NZ",
        "canonical_origin": "https://www.legislation.govt.nz/act/public/1986/0121/latest/DLM96439.html",
        "authority_classification": "PRIMARY_LEGISLATION",
        "reuse_terms": "Crown copyright (CC BY 4.0)",
        "expected_update_cadence": "ADHOC",
    }

    # Maintainer creates source
    res = await app_client.post(
        "/api/v1/policy/sources",
        json=payload,
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["id"] == "nz-legislation-fta-1986"
    assert data["title"] == "Fair Trading Act 1986"
    assert data["authority_classification"] == "PRIMARY_LEGISLATION"

    # Anyone authenticated can retrieve it
    get_res = await app_client.get(
        "/api/v1/policy/sources/nz-legislation-fta-1986",
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert get_res.status_code == 200
    assert get_res.json()["id"] == "nz-legislation-fta-1986"


@pytest.mark.asyncio
async def test_non_maintainer_cannot_create_policy_source(app_client: AsyncClient) -> None:
    """Requesters and unauthenticated callers are forbidden from creating policy sources."""
    payload = {
        "id": "nz-cga-1993",
        "title": "Consumer Guarantees Act 1993",
        "issuing_authority": "Parliament of New Zealand",
        "jurisdiction": "NZ",
        "canonical_origin": "https://www.legislation.govt.nz/act/public/1993/0091/latest/DLM311053.html",
        "authority_classification": "PRIMARY_LEGISLATION",
        "reuse_terms": "CC BY 4.0",
        "expected_update_cadence": "ADHOC",
    }

    # Requester forbidden
    res = await app_client.post(
        "/api/v1/policy/sources",
        json=payload,
        headers={"Authorization": "Bearer dev-requester-token"},
    )
    assert res.status_code == 403

    # Unauthenticated unauthorized
    res_no_auth = await app_client.post("/api/v1/policy/sources", json=payload)
    assert res_no_auth.status_code == 401


@pytest.mark.asyncio
async def test_maintainer_ingests_snapshot_idempotently(app_client: AsyncClient) -> None:
    """Ingesting the same content for a source returns the existing snapshot without duplicate error."""
    # 1. Register source
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

    raw_markdown = """# Fair Trading Act 1986
## Section 9: Misleading conduct
No person shall engage in misleading conduct in trade.
"""

    # 2. Ingest snapshot
    res1 = await app_client.post(
        "/api/v1/policy/sources/nz-fta-1986/snapshots",
        json={"raw_content": raw_markdown},
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res1.status_code == 201, res1.text
    snap1 = res1.json()
    assert snap1["source_id"] == "nz-fta-1986"
    assert len(snap1["passages"]) == 1
    assert snap1["passages"][0]["section_identifier"] == "Section 9"

    # 3. Ingest same snapshot again (idempotent)
    res2 = await app_client.post(
        "/api/v1/policy/sources/nz-fta-1986/snapshots",
        json={"raw_content": raw_markdown},
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert res2.status_code == 200 or res2.status_code == 201
    snap2 = res2.json()
    assert snap2["id"] == snap1["id"]
    assert snap2["content_hash"] == snap1["content_hash"]

    # 4. Fetch snapshot by ID
    res_get = await app_client.get(
        f"/api/v1/policy/snapshots/{snap1['id']}",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert res_get.status_code == 200
    assert res_get.json()["id"] == snap1["id"]

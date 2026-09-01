"""Tests for Assessment intake API routes, authorization, rate limiting, and safe errors."""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.api.deps import intake_rate_limiter
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.persistence.models import Base

TEST_DB_URL = "postgresql+psycopg://postgres:postgres@localhost:54329/postgres"


@pytest_asyncio.fixture
async def app_client() -> AsyncIterator[AsyncClient]:
    """Provide an AsyncClient for FastAPI application with test settings."""
    intake_rate_limiter.reset()
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
    intake_rate_limiter.reset()


@pytest.mark.asyncio
async def test_create_assessment_authorized(app_client: AsyncClient) -> None:
    """Verify authorized REQUESTER can create an assessment."""
    headers = {
        "Authorization": "Bearer dev-requester-token",
        "Idempotency-Key": "req-key-01",
    }
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {
            "sale_type": "DEALER",
            "intended_use": "Family car in Wellington",
            "questions": ["Is there any outstanding finance?"],
        },
    }

    response = await app_client.post("/api/v1/assessments", json=payload, headers=headers)
    assert response.status_code == 201
    data = response.json()
    assert data["vin"] == "1HGCR2F85HA000000"
    assert data["lifecycle_state"] == "IN_PROGRESS"
    assert data["current_run_number"] == 1
    assert "id" in data


@pytest.mark.asyncio
async def test_owner_read_assessment(app_client: AsyncClient) -> None:
    """Verify creator can read their own assessment."""
    headers = {
        "Authorization": "Bearer dev-requester-token",
        "Idempotency-Key": "req-key-owner",
    }
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "PRIVATE"},
    }

    create_resp = await app_client.post("/api/v1/assessments", json=payload, headers=headers)
    assert create_resp.status_code == 201
    assessment_id = create_resp.json()["id"]

    # Owner reads
    get_resp = await app_client.get(f"/api/v1/assessments/{assessment_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == assessment_id
    assert get_resp.json()["vin"] == "1HGCR2F85HA000000"


@pytest.mark.asyncio
async def test_reviewer_cannot_create_assessment(app_client: AsyncClient) -> None:
    """Verify REVIEWER role is forbidden from creating assessments (least privilege)."""
    headers = {
        "Authorization": "Bearer dev-reviewer-token",
        "Idempotency-Key": "rev-forbidden",
    }
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "DEALER"},
    }

    response = await app_client.post("/api/v1/assessments", json=payload, headers=headers)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_unauthorized_missing_token(app_client: AsyncClient) -> None:
    """Verify request without valid Authorization header returns 401."""
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "DEALER"},
    }
    response = await app_client.post(
        "/api/v1/assessments",
        json=payload,
        headers={"Idempotency-Key": "no-auth"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
async def test_missing_idempotency_key_rejected(app_client: AsyncClient) -> None:
    """Verify request without Idempotency-Key header is rejected."""
    headers = {"Authorization": "Bearer dev-requester-token"}
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "DEALER"},
    }
    response = await app_client.post("/api/v1/assessments", json=payload, headers=headers)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_IDEMPOTENCY_KEY"


@pytest.mark.asyncio
async def test_idempotent_conflict_on_different_payload(app_client: AsyncClient) -> None:
    """Verify reuse of idempotency key with different payload returns 409 CONFLICT."""
    headers = {
        "Authorization": "Bearer dev-requester-token",
        "Idempotency-Key": "conflict-key-api",
    }
    payload1 = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "DEALER"},
    }
    payload2 = {
        "vin": "1HGCR2F8XHA000008",
        "context": {"sale_type": "PRIVATE"},
    }

    r1 = await app_client.post("/api/v1/assessments", json=payload1, headers=headers)
    assert r1.status_code == 201

    r2 = await app_client.post("/api/v1/assessments", json=payload2, headers=headers)
    assert r2.status_code == 409
    assert r2.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.asyncio
async def test_rate_limiting_intake(app_client: AsyncClient) -> None:
    """Verify intake rate limiting bounds burst creation requests."""
    headers_base = {"Authorization": "Bearer dev-requester-token"}
    payload = {
        "vin": "1HGCR2F85HA000000",
        "context": {"sale_type": "DEALER"},
    }

    # Burst requests with distinct idempotency keys
    responses = []
    for i in range(25):
        h = {**headers_base, "Idempotency-Key": f"burst-key-{i}"}
        resp = await app_client.post("/api/v1/assessments", json=payload, headers=h)
        responses.append(resp)

    # Some requests in the burst should trigger 429 Too Many Requests
    status_codes = [r.status_code for r in responses]
    assert 429 in status_codes
    rate_limited = [r for r in responses if r.status_code == 429][0]
    assert rate_limited.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_operator_and_maintainer_forbidden_from_reading_assessment(
    app_client: AsyncClient,
) -> None:
    """Verify TECHNICAL_OPERATOR and POLICY_CORPUS_MAINTAINER are forbidden from reading assessments."""
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-read-auth"},
    )
    assert create_resp.status_code == 201
    assessment_id = create_resp.json()["id"]

    # Technical operator probe
    op_resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}",
        headers={"Authorization": "Bearer dev-operator-token"},
    )
    assert op_resp.status_code == 403
    assert op_resp.json()["error"]["code"] == "FORBIDDEN"

    # Maintainer probe
    maint_resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}",
        headers={"Authorization": "Bearer dev-maintainer-token"},
    )
    assert maint_resp.status_code == 403
    assert maint_resp.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.asyncio
async def test_reviewer_can_read_assessment(app_client: AsyncClient) -> None:
    """Verify REVIEWER role can read assessments for review purposes."""
    create_resp = await app_client.post(
        "/api/v1/assessments",
        json={"vin": "1HGCR2F85HA000000", "context": {"sale_type": "DEALER"}},
        headers={"Authorization": "Bearer dev-requester-token", "Idempotency-Key": "req-rev-read"},
    )
    assert create_resp.status_code == 201
    assessment_id = create_resp.json()["id"]

    rev_resp = await app_client.get(
        f"/api/v1/assessments/{assessment_id}",
        headers={"Authorization": "Bearer dev-reviewer-token"},
    )
    assert rev_resp.status_code == 200
    assert rev_resp.json()["id"] == assessment_id

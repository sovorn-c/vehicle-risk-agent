"""Tests for Risk Policy creation, validation, activation, and inspection API routes."""

import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.persistence.models import Base, RiskPolicyRecord

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

    if hasattr(app.state, "engine") and app.state.engine is not None:
        await app.state.engine.dispose()
    await engine.dispose()


@pytest.mark.asyncio
async def test_operator_creates_validates_and_activates_risk_policy(
    app_client: AsyncClient,
) -> None:
    """Technical Operator creates DRAFT policy, advances to READY, and activates policy."""
    operator_headers = {"Authorization": "Bearer dev-operator-token"}

    # 1. Create policy (DRAFT)
    res_create = await app_client.post(
        "/api/v1/risk-policies",
        json={
            "id": "nz-risk-policy-v1",
            "name": "New Zealand Risk Policy v1",
            "description": "Baseline policy for NZ vehicle risk scoring",
            "version": "v1",
        },
        headers=operator_headers,
    )
    assert res_create.status_code == 201
    created_data = res_create.json()
    assert created_data["id"] == "nz-risk-policy-v1"
    assert created_data["lifecycle_state"] == "DRAFT"
    assert created_data["factor_weights"]["MATCH"] == 30
    assert created_data["factor_weights"]["LISTED"] == 45
    assert created_data["factor_weights"]["REPAIRABLE"] == 20
    assert created_data["factor_weights"]["STATUTORY"] == 40
    assert len(created_data["risk_bands"]) == 4

    # 2. Advance to READY
    res_ready = await app_client.post(
        "/api/v1/risk-policies/nz-risk-policy-v1/ready",
        headers=operator_headers,
    )
    assert res_ready.status_code == 200
    ready_data = res_ready.json()
    assert ready_data["lifecycle_state"] == "READY"

    # 3. Activate policy
    res_activate = await app_client.post(
        "/api/v1/risk-policies/nz-risk-policy-v1/activate",
        headers=operator_headers,
    )
    assert res_activate.status_code == 200
    act_data = res_activate.json()
    assert act_data["active_policy"]["id"] == "nz-risk-policy-v1"
    assert act_data["active_policy"]["lifecycle_state"] == "ACTIVE"
    assert act_data["active_policy"]["activated_at"] is not None
    assert act_data["retired_policy"] is None

    # 4. Get active policy
    res_active = await app_client.get(
        "/api/v1/risk-policies/active",
        headers=operator_headers,
    )
    assert res_active.status_code == 200
    assert res_active.json()["id"] == "nz-risk-policy-v1"
    assert res_active.json()["lifecycle_state"] == "ACTIVE"


@pytest.mark.asyncio
async def test_activating_new_policy_retires_prior_active_policy(app_client: AsyncClient) -> None:
    """Activating Policy 2 transitions Policy 1 to RETIRED while preserving replay access."""
    operator_headers = {"Authorization": "Bearer dev-operator-token"}

    # Create and activate Policy 1
    await app_client.post(
        "/api/v1/risk-policies",
        json={"id": "policy-1", "name": "Policy 1", "description": "First policy"},
        headers=operator_headers,
    )
    await app_client.post("/api/v1/risk-policies/policy-1/ready", headers=operator_headers)
    await app_client.post("/api/v1/risk-policies/policy-1/activate", headers=operator_headers)

    # Create and activate Policy 2
    await app_client.post(
        "/api/v1/risk-policies",
        json={"id": "policy-2", "name": "Policy 2", "description": "Second policy"},
        headers=operator_headers,
    )
    await app_client.post("/api/v1/risk-policies/policy-2/ready", headers=operator_headers)
    res_act2 = await app_client.post(
        "/api/v1/risk-policies/policy-2/activate", headers=operator_headers
    )
    assert res_act2.status_code == 200
    act2_data = res_act2.json()

    # Verify active and retired policies in response
    assert act2_data["active_policy"]["id"] == "policy-2"
    assert act2_data["active_policy"]["lifecycle_state"] == "ACTIVE"
    assert act2_data["retired_policy"]["id"] == "policy-1"
    assert act2_data["retired_policy"]["lifecycle_state"] == "RETIRED"
    assert act2_data["retired_policy"]["retired_at"] is not None

    # Inspect retired Policy 1 by ID for audit and replay
    res_p1 = await app_client.get("/api/v1/risk-policies/policy-1", headers=operator_headers)
    assert res_p1.status_code == 200
    assert res_p1.json()["id"] == "policy-1"
    assert res_p1.json()["lifecycle_state"] == "RETIRED"
    assert res_p1.json()["retired_at"] is not None

    # Active endpoint returns Policy 2
    res_active = await app_client.get("/api/v1/risk-policies/active", headers=operator_headers)
    assert res_active.status_code == 200
    assert res_active.json()["id"] == "policy-2"


@pytest.mark.asyncio
async def test_non_operator_roles_forbidden_from_policy_mutations(app_client: AsyncClient) -> None:
    """Requesters, Reviewers, and Policy Maintainers cannot mutate risk policies (403 Forbidden)."""
    forbidden_tokens = [
        "dev-requester-token",
        "dev-reviewer-token",
        "dev-maintainer-token",
    ]

    for token in forbidden_tokens:
        headers = {"Authorization": f"Bearer {token}"}

        # Create forbidden
        res_create = await app_client.post(
            "/api/v1/risk-policies",
            json={"id": f"policy-{token}", "name": "Unauthorized Policy", "description": "Test"},
            headers=headers,
        )
        assert res_create.status_code == 403

        # Ready forbidden
        res_ready = await app_client.post(
            "/api/v1/risk-policies/test-pol/ready",
            headers=headers,
        )
        assert res_ready.status_code == 403

        # Activate forbidden
        res_act = await app_client.post(
            "/api/v1/risk-policies/test-pol/activate",
            headers=headers,
        )
        assert res_act.status_code == 403


@pytest.mark.asyncio
async def test_unauthenticated_request_returns_401(app_client: AsyncClient) -> None:
    """Requests with missing or invalid tokens receive 401 Unauthorized."""
    # Missing header
    res_none = await app_client.get("/api/v1/risk-policies/active")
    assert res_none.status_code == 401

    # Invalid token
    res_invalid = await app_client.get(
        "/api/v1/risk-policies/active",
        headers={"Authorization": "Bearer invalid-secret-token"},
    )
    assert res_invalid.status_code == 401


@pytest.mark.asyncio
async def test_invalid_lifecycle_transitions_rejected(app_client: AsyncClient) -> None:
    """Activating unready DRAFT policy or non-existent policy is rejected."""
    operator_headers = {"Authorization": "Bearer dev-operator-token"}

    # 1. Create policy in DRAFT state
    await app_client.post(
        "/api/v1/risk-policies",
        json={"id": "policy-draft-test", "name": "Draft Policy", "description": "Test"},
        headers=operator_headers,
    )

    # 2. Attempt direct activation without marking READY (400 Bad Request)
    res_direct = await app_client.post(
        "/api/v1/risk-policies/policy-draft-test/activate",
        headers=operator_headers,
    )
    assert res_direct.status_code == 400
    assert res_direct.json()["error"]["code"] == "INVALID_STATE_TRANSITION"

    # 3. Ready and activate
    await app_client.post("/api/v1/risk-policies/policy-draft-test/ready", headers=operator_headers)
    await app_client.post(
        "/api/v1/risk-policies/policy-draft-test/activate", headers=operator_headers
    )

    # 4. Attempting to activate already active policy returns 409 or 400
    res_re_act = await app_client.post(
        "/api/v1/risk-policies/policy-draft-test/activate",
        headers=operator_headers,
    )
    assert res_re_act.status_code in (400, 409)

    # 5. Non-existent policy returns 404
    res_404 = await app_client.post(
        "/api/v1/risk-policies/non-existent-pol/activate",
        headers=operator_headers,
    )
    assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_read_endpoints_accessible_to_all_authenticated_roles(
    app_client: AsyncClient,
) -> None:
    """GET endpoints are accessible to Requesters, Reviewers, Maintainers, and Operators."""
    operator_headers = {"Authorization": "Bearer dev-operator-token"}

    # Setup active policy
    await app_client.post(
        "/api/v1/risk-policies",
        json={"id": "policy-read-test", "name": "Read Test Policy", "description": "Test"},
        headers=operator_headers,
    )
    await app_client.post("/api/v1/risk-policies/policy-read-test/ready", headers=operator_headers)
    await app_client.post(
        "/api/v1/risk-policies/policy-read-test/activate", headers=operator_headers
    )

    roles_tokens = [
        "dev-requester-token",
        "dev-reviewer-token",
        "dev-maintainer-token",
        "dev-operator-token",
    ]

    for token in roles_tokens:
        headers = {"Authorization": f"Bearer {token}"}
        res_list = await app_client.get("/api/v1/risk-policies", headers=headers)
        assert res_list.status_code == 200
        assert len(res_list.json()) >= 1

        res_active = await app_client.get("/api/v1/risk-policies/active", headers=headers)
        assert res_active.status_code == 200
        assert res_active.json()["id"] == "policy-read-test"

        res_single = await app_client.get("/api/v1/risk-policies/policy-read-test", headers=headers)
        assert res_single.status_code == 200
        assert res_single.json()["id"] == "policy-read-test"


@pytest.mark.asyncio
async def test_concurrent_activation_enforces_single_active_invariant(
    app_client: AsyncClient,
) -> None:
    """Enforce that concurrent activation requests maintain strictly one active policy."""
    operator_headers = {"Authorization": "Bearer dev-operator-token"}

    # Create policies A, B, C and mark READY
    for name in ["A", "B", "C"]:
        await app_client.post(
            "/api/v1/risk-policies",
            json={
                "id": f"policy-concurrent-{name}",
                "name": f"Policy {name}",
                "description": "Test",
            },
            headers=operator_headers,
        )
        await app_client.post(
            f"/api/v1/risk-policies/policy-concurrent-{name}/ready", headers=operator_headers
        )

    # Concurrently activate A, B, C
    tasks = [
        app_client.post(
            f"/api/v1/risk-policies/policy-concurrent-{name}/activate", headers=operator_headers
        )
        for name in ["A", "B", "C"]
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    assert len(responses) == 3

    # Verify at least one succeeded and others either succeeded in sequence or handled cleanly
    engine = create_async_engine(TEST_DB_URL, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as db_sess:
        stmt = select(RiskPolicyRecord).where(RiskPolicyRecord.lifecycle_state == "ACTIVE")
        active_records = (await db_sess.execute(stmt)).scalars().all()
        assert len(active_records) == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_activate_policy_lexicographical_precedence(app_client: AsyncClient) -> None:
    """Activating policy with lower ID (pol-aaa) when pol-zzz is active succeeds."""
    operator_headers = {"Authorization": "Bearer dev-operator-token"}

    # 1. Create and activate pol-zzz
    await app_client.post(
        "/api/v1/risk-policies",
        json={"id": "pol-zzz", "name": "Policy Z", "description": "Active Policy Z"},
        headers=operator_headers,
    )
    await app_client.post("/api/v1/risk-policies/pol-zzz/ready", headers=operator_headers)
    res_z = await app_client.post(
        "/api/v1/risk-policies/pol-zzz/activate",
        headers=operator_headers,
    )
    assert res_z.status_code == 200

    # 2. Create and activate pol-aaa (sorts lexicographically before pol-zzz)
    await app_client.post(
        "/api/v1/risk-policies",
        json={"id": "pol-aaa", "name": "Policy A", "description": "New Policy A"},
        headers=operator_headers,
    )
    await app_client.post("/api/v1/risk-policies/pol-aaa/ready", headers=operator_headers)
    res_a = await app_client.post(
        "/api/v1/risk-policies/pol-aaa/activate",
        headers=operator_headers,
    )
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["active_policy"]["id"] == "pol-aaa"
    assert data_a["active_policy"]["lifecycle_state"] == "ACTIVE"
    assert data_a["retired_policy"]["id"] == "pol-zzz"
    assert data_a["retired_policy"]["lifecycle_state"] == "RETIRED"

"""Opt-in end-to-end smoke for the sourced E12 corpus and live retrieval path."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from tests.database import TEST_DB_URL
from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.persistence.models import Base

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        os.getenv("RUN_E12_CORPUS_SMOKE") != "1",
        reason="Set RUN_E12_CORPUS_SMOKE=1 to run the live corpus smoke test",
    ),
]


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    reset_engine = create_async_engine(TEST_DB_URL, echo=False)
    async with reset_engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    manifest_path = Path(__file__).parents[2] / "docs/benchmarks/e12/corpus/manifest.json"
    retrieval_config = json.loads(manifest_path.read_text(encoding="utf-8"))["retrieval_config"]
    app = create_app(
        Settings(
            database_url=TEST_DB_URL,
            retrieval_mode="live",
            embedding_revision=retrieval_config["embedding_revision"],
            reranker_revision=retrieval_config["reranker_revision"],
        )
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as test_client:
        yield test_client

    await app.state.engine.dispose()
    await reset_engine.dispose()


@pytest.mark.asyncio
async def test_portfolio_corpus_activates_and_retrieves_live_passages(
    client: AsyncClient,
) -> None:
    manifest_path = Path(__file__).parents[2] / "docs/benchmarks/e12/corpus/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    headers = {"Authorization": "Bearer dev-maintainer-token"}
    snapshot_ids: list[str] = []

    for source in manifest["sources"]:
        create_source = {
            key: source[key]
            for key in (
                "id",
                "title",
                "issuing_authority",
                "canonical_origin",
                "authority_classification",
                "reuse_terms",
                "expected_update_cadence",
            )
        }
        source_response = await client.post(
            "/api/v1/policy/sources", json=create_source, headers=headers
        )
        assert source_response.status_code == 201, source_response.text

        snapshot_response = await client.post(
            f"/api/v1/policy/sources/{source['id']}/snapshots",
            json={"raw_content": source["content"]},
            headers=headers,
        )
        assert snapshot_response.status_code == 201, snapshot_response.text
        snapshot = snapshot_response.json()
        assert snapshot["content_hash"] == source["content_sha256"]
        assert snapshot["passages"]
        snapshot_ids.append(snapshot["id"])

    corpus_id = manifest["corpus_id"]
    create_corpus = await client.post(
        "/api/v1/policy/corpora",
        json={
            "id": corpus_id,
            "name": manifest["name"],
            "description": manifest["description"],
            "snapshot_ids": snapshot_ids,
            "retrieval_config": manifest["retrieval_config"],
        },
        headers=headers,
    )
    assert create_corpus.status_code == 201, create_corpus.text

    ready = await client.post(f"/api/v1/policy/corpora/{corpus_id}/ready", headers=headers)
    assert ready.status_code == 200, ready.text
    assert ready.json()["lifecycle_state"] == "READY"

    activated = await client.post(f"/api/v1/policy/corpora/{corpus_id}/activate", headers=headers)
    assert activated.status_code == 200, activated.text
    assert activated.json()["active_corpus"]["lifecycle_state"] == "ACTIVE"

    response = await client.post(
        "/api/v1/policy/retrieve",
        headers={"Authorization": "Bearer dev-requester-token"},
        json={"query": "What is the acceptable quality guarantee for a used vehicle?"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["is_abstention"] is False
    assert any(
        citation["source_id"] == "nz-consumer-guarantees-act-1993"
        for citation in result["citations"]
    )

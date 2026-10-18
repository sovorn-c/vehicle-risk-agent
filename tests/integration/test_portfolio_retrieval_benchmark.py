"""Opt-in PostgreSQL and pinned-model run for the frozen E12 benchmark."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from tests.database import TEST_DB_URL
from vehicle_risk_agent.api.app import create_app
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.evaluation.portfolio_corpus import load_portfolio_corpus
from vehicle_risk_agent.evaluation.portfolio_retrieval import load_portfolio_retrieval_dataset
from vehicle_risk_agent.evaluation.portfolio_retrieval_runner import (
    create_run_directory,
    run_portfolio_retrieval_comparison,
    write_portfolio_retrieval_run,
)
from vehicle_risk_agent.persistence.models import Base

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        os.getenv("RUN_E12_RETRIEVAL_BENCHMARK") != "1",
        reason="Set RUN_E12_RETRIEVAL_BENCHMARK=1 to execute the live retrieval benchmark",
    ),
]


@pytest_asyncio.fixture
async def benchmark_client() -> AsyncIterator[tuple[AsyncClient, FastAPI]]:
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
        yield test_client, app

    await app.state.engine.dispose()
    await reset_engine.dispose()


@pytest.mark.asyncio
async def test_portfolio_retrieval_benchmark_runs_and_exports_every_evaluation_query(
    benchmark_client: tuple[AsyncClient, FastAPI],
) -> None:
    client, app = benchmark_client
    root = Path(__file__).parents[2]
    corpus = load_portfolio_corpus(root / "docs/benchmarks/e12/corpus/manifest.json")
    dataset = load_portfolio_retrieval_dataset(
        root / "docs/benchmarks/e12/retrieval/dataset-v1.json", corpus
    )
    source_manifest = json.loads(
        (root / "docs/benchmarks/e12/corpus/manifest.json").read_text(encoding="utf-8")
    )
    auth = {"Authorization": f"Bearer {app.state.settings.maintainer_token}"}

    for source in source_manifest["sources"]:
        source_payload = {
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
        created = await client.post("/api/v1/policy/sources", json=source_payload, headers=auth)
        assert created.status_code == 201, created.text
        ingested = await client.post(
            f"/api/v1/policy/sources/{source['id']}/snapshots",
            json={"raw_content": source["content"]},
            headers=auth,
        )
        assert ingested.status_code == 201, ingested.text
        assert ingested.json()["content_hash"] == source["content_sha256"]

    created_corpus = await client.post(
        "/api/v1/policy/corpora",
        json={
            "id": source_manifest["corpus_id"],
            "name": source_manifest["name"],
            "description": source_manifest["description"],
            "snapshot_ids": list(corpus.snapshot_ids),
            "retrieval_config": source_manifest["retrieval_config"],
        },
        headers=auth,
    )
    assert created_corpus.status_code == 201, created_corpus.text
    corpus_id = source_manifest["corpus_id"]
    ready = await client.post(f"/api/v1/policy/corpora/{corpus_id}/ready", headers=auth)
    assert ready.status_code == 200, ready.text
    active = await client.post(f"/api/v1/policy/corpora/{corpus_id}/activate", headers=auth)
    assert active.status_code == 200, active.text

    session_factory = app.state.session_factory
    async with session_factory() as session:
        results, summaries = await run_portfolio_retrieval_comparison(session, corpus, dataset)

    assert set(results) == {"keyword", "hybrid"}
    assert all(len(scores) == 20 for scores in results.values())
    assert all(summary.expected_query_count == 20 for summary in summaries.values())

    output_directory = create_run_directory(root / "docs/benchmarks/e12/retrieval/runs")
    json_path, csv_path = write_portfolio_retrieval_run(
        output_directory, corpus, dataset, results, summaries
    )
    run_document = json.loads(json_path.read_text(encoding="utf-8"))
    assert run_document["execution_status"] in {"complete", "partial"}
    assert run_document["corpus_fingerprint"] == corpus.fingerprint
    assert len(run_document["methods"]["keyword"]["queries"]) == 20
    assert len(run_document["methods"]["hybrid"]["queries"]) == 20
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 41

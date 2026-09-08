"""Tests for Ragas-compatible retrieval and faithfulness record export."""

# story: e06s03
# task: e06s03-t03

import json

import pytest

from vehicle_risk_agent.evaluation.retrieval import (
    build_seeded_retrieval_service,
    export_ragas_records,
    get_seeded_retrieval_dataset,
)


@pytest.mark.asyncio
async def test_ragas_export_matches_schema_contract() -> None:
    """Exported records contain user_input, retrieved_contexts, response, and reference."""
    dataset = get_seeded_retrieval_dataset()
    service = await build_seeded_retrieval_service(dataset)

    retrieval_results = {}
    for q in dataset.queries:
        retrieval_results[q.query_id] = await service.retrieve(q.query)

    records = export_ragas_records(
        dataset=dataset,
        retrieval_results=retrieval_results,
    )

    assert len(records) == len(dataset.queries)

    required_keys = {
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
        "reference_contexts",
    }
    for record in records:
        assert required_keys.issubset(record.keys())
        assert isinstance(record["user_input"], str)
        assert isinstance(record["retrieved_contexts"], list)
        assert isinstance(record["response"], str)
        assert isinstance(record["reference"], str)
        assert isinstance(record["reference_contexts"], list)
        assert len(record["user_input"]) > 0


@pytest.mark.asyncio
async def test_ragas_export_abstention_handling() -> None:
    """No-answer queries export empty retrieved contexts and explicit abstention response."""
    dataset = get_seeded_retrieval_dataset()
    service = await build_seeded_retrieval_service(dataset)

    retrieval_results = {}
    for q in dataset.queries:
        retrieval_results[q.query_id] = await service.retrieve(q.query)

    records = export_ragas_records(
        dataset=dataset,
        retrieval_results=retrieval_results,
    )

    no_answer_records = [
        r for r, q in zip(records, dataset.queries, strict=False) if q.is_no_answer
    ]
    assert len(no_answer_records) == 2

    for r in no_answer_records:
        assert len(r["retrieved_contexts"]) == 0
        assert len(r["reference_contexts"]) == 0
        assert "no" in r["response"].lower() or "abstain" in r["response"].lower()


@pytest.mark.asyncio
async def test_ragas_export_security_isolation() -> None:
    """Export records contain no credentials, raw prompts, or internal stack traces."""
    dataset = get_seeded_retrieval_dataset()
    service = await build_seeded_retrieval_service(dataset)

    retrieval_results = {}
    for q in dataset.queries:
        retrieval_results[q.query_id] = await service.retrieve(q.query)

    records = export_ragas_records(
        dataset=dataset,
        retrieval_results=retrieval_results,
    )

    dump = json.dumps(records)
    for sensitive in ("sk-ant", "api_key", "bearer", "password", "secret", "system_prompt"):
        assert sensitive not in dump.lower()

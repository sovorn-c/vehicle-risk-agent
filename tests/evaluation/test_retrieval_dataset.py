"""Tests for policy retrieval evaluation dataset and labels."""

# story: e06s03

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.evaluation.retrieval import (
    RetrievalEvaluationDataset,
    RetrievalQueryLabel,
    get_seeded_policy_passages,
    get_seeded_retrieval_dataset,
)


def test_retrieval_query_label_validation() -> None:
    label = RetrievalQueryLabel(
        query_id="q-001",
        query="What are the penalties for false odometer representations?",
        relevant_passage_ids=("snap-fta:p002",),
        required_citation_ids=("snap-fta:p002",),
        is_no_answer=False,
        intent="Odometer tampering under Fair Trading Act",
    )
    assert label.query_id == "q-001"
    assert label.is_no_answer is False

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        RetrievalQueryLabel(
            query_id="q-001",
            query="test",
            relevant_passage_ids=(),
            is_no_answer=True,
            extra_field="disallowed",  # type: ignore[call-arg]
        )


def test_seeded_policy_passages_and_dataset_integrity() -> None:
    passages = get_seeded_policy_passages()
    assert len(passages) >= 5
    passage_ids = {p.id for p in passages}

    dataset = get_seeded_retrieval_dataset()
    assert isinstance(dataset, RetrievalEvaluationDataset)
    assert len(dataset.queries) >= 10
    assert len(dataset.passages) >= 5

    no_answer_count = 0
    answered_count = 0

    for q in dataset.queries:
        if q.is_no_answer:
            no_answer_count += 1
            msg = f"No-answer query {q.query_id} has relevant passages"
            assert len(q.relevant_passage_ids) == 0, msg
        else:
            answered_count += 1
            msg = f"Answered query {q.query_id} has no relevant passages"
            assert len(q.relevant_passage_ids) > 0, msg
            for pid in q.relevant_passage_ids:
                msg = f"Passage {pid} in query {q.query_id} not found in passages"
                assert pid in passage_ids, msg

    assert no_answer_count >= 2, f"Expected at least 2 no-answer queries, got {no_answer_count}"
    assert answered_count >= 8, f"Expected at least 8 answered queries, got {answered_count}"

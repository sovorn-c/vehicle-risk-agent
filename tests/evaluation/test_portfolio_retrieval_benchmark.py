from pathlib import Path

from vehicle_risk_agent.evaluation.portfolio_retrieval import (
    load_portfolio_retrieval_dataset,
)

from vehicle_risk_agent.evaluation.portfolio_corpus import load_portfolio_corpus


def test_portfolio_retrieval_dataset_is_frozen_and_corpus_bound() -> None:
    root = Path(__file__).parents[2]
    corpus = load_portfolio_corpus(root / "docs/benchmarks/e12/corpus/manifest.json")
    dataset = load_portfolio_retrieval_dataset(
        root / "docs/benchmarks/e12/retrieval/dataset-v1.json", corpus
    )

    development = [query for query in dataset.queries if query.split == "development"]
    evaluation = [query for query in dataset.queries if query.split == "evaluation"]
    answerable = [query for query in evaluation if query.relevant_passage_ids]
    unsupported = [query for query in evaluation if not query.relevant_passage_ids]
    corpus_passage_ids = {passage.id for passage in corpus.passages}

    assert dataset.corpus_fingerprint == corpus.fingerprint
    assert len(development) == 6
    assert len(evaluation) == 20
    assert len(answerable) == 15
    assert len(unsupported) == 5
    assert {query.id for query in development}.isdisjoint(query.id for query in evaluation)
    assert all(query.relevant_passage_ids for query in development if query.expected_proposition)
    assert all(query.relevant_passage_ids for query in answerable)
    assert all(query.unsupported_reason for query in unsupported)
    assert all(
        passage_id in corpus_passage_ids
        for query in dataset.queries
        for passage_id in query.relevant_passage_ids
    )
    assert dataset.top_k == 5
    assert len(dataset.fingerprint) == 64

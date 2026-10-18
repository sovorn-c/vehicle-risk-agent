from pathlib import Path

from vehicle_risk_agent.evaluation.portfolio_retrieval import (
    PortfolioRetrievalQuery,
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



def test_portfolio_retrieval_query_metrics_use_fixed_slots_and_separate_failures() -> None:
    from vehicle_risk_agent.evaluation.portfolio_retrieval import score_portfolio_query

    answerable = PortfolioRetrievalQuery(
        id="metric-supported",
        split="evaluation",
        query="Find the labeled passages.",
        relevant_passage_ids=("p1", "p2"),
        expected_proposition="Both passages support the proposition.",
    )
    scored = score_portfolio_query(
        answerable,
        retrieved_passage_ids=["p3", "p2", "p1"],
        citation_passage_ids=["p3", "p2", "p1"],
        is_abstention=False,
        known_passage_ids={"p1", "p2", "p3"},
    )

    assert scored.recall_at_5 == 1.0
    assert scored.precision_at_5 == 0.4
    assert scored.reciprocal_rank_at_5 == 0.5
    assert scored.false_abstention is False
    assert scored.citation_identity_valid is True

    unsupported = PortfolioRetrievalQuery(
        id="metric-unsupported",
        split="evaluation",
        query="Ask for an absent topic.",
        unsupported_reason="The topic is not present in the corpus.",
    )
    abstained = score_portfolio_query(
        unsupported,
        retrieved_passage_ids=[],
        citation_passage_ids=[],
        is_abstention=True,
        known_passage_ids={"p1", "p2", "p3"},
    )
    assert abstained.correct_abstention is True

    failed = score_portfolio_query(
        unsupported,
        retrieved_passage_ids=[],
        citation_passage_ids=[],
        is_abstention=None,
        known_passage_ids={"p1", "p2", "p3"},
        technical_error="database unavailable",
    )
    assert failed.technical_failure is True
    assert failed.correct_abstention is None
    assert failed.recall_at_5 is None

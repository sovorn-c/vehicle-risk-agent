from pathlib import Path

from vehicle_risk_agent.evaluation.portfolio_corpus import load_portfolio_corpus
from vehicle_risk_agent.evaluation.portfolio_retrieval import (
    PortfolioRetrievalQuery,
    load_portfolio_retrieval_dataset,
)


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
    assert failed.technical_error is not None
    assert failed.correct_abstention is None
    assert failed.recall_at_5 is None



def test_portfolio_retrieval_summary_requires_complete_evaluation_denominators() -> None:
    from vehicle_risk_agent.evaluation.portfolio_retrieval import (
        load_portfolio_retrieval_dataset,
        score_portfolio_query,
        summarize_portfolio_results,
    )

    root = Path(__file__).parents[2]
    corpus = load_portfolio_corpus(root / "docs/benchmarks/e12/corpus/manifest.json")
    dataset = load_portfolio_retrieval_dataset(
        root / "docs/benchmarks/e12/retrieval/dataset-v1.json", corpus
    )
    known_passage_ids = {passage.id for passage in corpus.passages}
    evaluation_queries = [query for query in dataset.queries if query.split == "evaluation"]
    scores = [
        score_portfolio_query(
            query,
            retrieved_passage_ids=query.relevant_passage_ids,
            citation_passage_ids=query.relevant_passage_ids,
            is_abstention=not query.relevant_passage_ids,
            known_passage_ids=known_passage_ids,
        )
        for query in evaluation_queries
    ]

    complete = summarize_portfolio_results("hybrid", dataset, scores)
    assert complete.execution_status == "complete"
    assert complete.answerable_denominator == 15
    assert complete.unsupported_denominator == 5
    assert complete.recall_at_5 == 1.0
    assert complete.mrr_at_5 == 1.0
    assert complete.precision_at_5 == dataset.precision_at_5_ceiling
    assert complete.unsupported_abstention_rate == 1.0
    assert complete.target_status == "met"

    partial = summarize_portfolio_results("hybrid", dataset, scores[:-1])
    assert partial.execution_status == "partial"
    assert partial.completed_query_count == 19
    assert partial.recall_at_5 is None
    assert partial.target_status == "not_evaluable"

    failed_score = score_portfolio_query(
        evaluation_queries[-1],
        retrieved_passage_ids=[],
        citation_passage_ids=[],
        is_abstention=None,
        known_passage_ids=known_passage_ids,
        technical_error="database unavailable",
    )
    failed = summarize_portfolio_results("hybrid", dataset, [*scores[:-1], failed_score])
    assert failed.technical_failure_count == 1
    assert failed.unsupported_correct_abstention_count == 4
    assert failed.unsupported_abstention_rate is None
    assert failed.target_status == "not_evaluable"



def test_empty_portfolio_retrieval_run_is_partial_not_a_pass() -> None:
    from vehicle_risk_agent.evaluation.portfolio_retrieval import summarize_portfolio_results

    root = Path(__file__).parents[2]
    corpus = load_portfolio_corpus(root / "docs/benchmarks/e12/corpus/manifest.json")
    dataset = load_portfolio_retrieval_dataset(
        root / "docs/benchmarks/e12/retrieval/dataset-v1.json", corpus
    )

    summary = summarize_portfolio_results("keyword", dataset, [])

    assert summary.execution_status == "partial"
    assert summary.expected_query_count == 20
    assert summary.completed_query_count == 0
    assert summary.target_status == "not_evaluable"

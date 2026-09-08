"""Tests for deterministic retrieval and grounding quality metrics."""

# story: e06s03
# task: e06s03-t02

import pytest

from vehicle_risk_agent.evaluation.retrieval import (
    RetrievalMetricsEvaluator,
    RetrievalMetricsResult,
    RetrievalMetricsThresholds,
    build_seeded_retrieval_service,
    get_seeded_retrieval_dataset,
)
from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex
from vehicle_risk_agent.retrieval.service import HybridRetrievalService, RetrievalResult


@pytest.mark.asyncio
async def test_seeded_corpus_meets_all_deterministic_thresholds() -> None:
    """The seeded policy corpus meets recall, precision, MRR, abstention, and citation gates."""
    dataset = get_seeded_retrieval_dataset()
    service = await build_seeded_retrieval_service(dataset)
    thresholds = RetrievalMetricsThresholds(
        context_recall_at_5_min=0.90,
        context_precision_at_5_min=0.70,
        mrr_min=0.80,
        abstention_accuracy_min=0.95,
        citation_identity_accuracy=1.00,
    )
    evaluator = RetrievalMetricsEvaluator(service=service, thresholds=thresholds)
    result = await evaluator.evaluate_dataset(dataset)

    assert isinstance(result, RetrievalMetricsResult)
    assert result.passed is True
    assert result.context_recall_at_5 >= thresholds.context_recall_at_5_min
    assert result.context_precision_at_5 >= thresholds.context_precision_at_5_min
    assert result.mrr >= thresholds.mrr_min
    assert result.abstention_accuracy >= thresholds.abstention_accuracy_min
    assert result.citation_identity_accuracy >= thresholds.citation_identity_accuracy
    assert len(result.run_hash) == 64
    assert len(result.query_metrics) == len(dataset.queries)


@pytest.mark.asyncio
async def test_failure_when_no_answer_query_does_not_abstain() -> None:
    """If a no-answer query retrieves non-empty citations, abstention drops and gate fails."""
    dataset = get_seeded_retrieval_dataset()

    class PermissiveReranker:
        async def rerank(self, query: str, texts: list[str]) -> list[float]:
            _ = query
            return [0.99] * len(texts)

    config = RetrievalConfiguration(final_passage_cap=5, minimum_reranker_score=0.1)
    index = InMemoryPolicyIndex(embedder=FakeEmbeddingAdapter(), config=config)
    await index.build_index(list(dataset.passages))

    source_metadata = {
        "nz-legislation-fta-1986": ("Fair Trading Act 1986", "https://legislation.govt.nz/fta"),
        "nz-legislation-cga-1993": (
            "Consumer Guarantees Act 1993",
            "https://legislation.govt.nz/cga",
        ),
        "nz-ppsr-act-1999": ("Personal Property Securities Act 1999", "https://ppsr.govt.nz/act"),
        "nzta-virm-manual": ("NZTA VIRM", "https://vehicleinspection.nzta.govt.nz/virm"),
    }
    service = HybridRetrievalService(
        index=index,
        reranker=PermissiveReranker(),
        config=config,
        source_metadata=source_metadata,
    )

    evaluator = RetrievalMetricsEvaluator(service=service)
    result = await evaluator.evaluate_dataset(dataset)

    assert result.abstention_accuracy < 0.95
    assert result.passed is False


@pytest.mark.asyncio
async def test_failure_when_citation_identity_is_missing() -> None:
    """If required citation identity is missing from retrieved citations, gate fails."""
    dataset = get_seeded_retrieval_dataset()

    class EmptyCitationService:
        async def retrieve(self, query: str) -> RetrievalResult:
            return RetrievalResult(
                query=query,
                dense_candidates=[],
                keyword_candidates=[],
                fused_candidates=[],
                reranked_candidates=[],
                citations=[],
                is_abstention=True,
            )

    evaluator = RetrievalMetricsEvaluator(service=EmptyCitationService())
    result = await evaluator.evaluate_dataset(dataset)

    assert result.citation_identity_accuracy == 0.0
    assert result.passed is False


@pytest.mark.asyncio
async def test_metrics_reproducibility_and_security() -> None:
    """Metrics computation is deterministic and contains no leaked credentials or prompts."""
    dataset = get_seeded_retrieval_dataset()
    service = await build_seeded_retrieval_service(dataset)
    evaluator = RetrievalMetricsEvaluator(service=service)

    result1 = await evaluator.evaluate_dataset(dataset)
    result2 = await evaluator.evaluate_dataset(dataset)

    assert result1.run_hash == result2.run_hash
    assert result1.context_recall_at_5 == result2.context_recall_at_5
    assert result1.context_precision_at_5 == result2.context_precision_at_5
    assert result1.mrr == result2.mrr

    dump = result1.model_dump_json()
    for sensitive in ("sk-ant", "password", "api_key", "secret", "authorization"):
        assert sensitive not in dump.lower()

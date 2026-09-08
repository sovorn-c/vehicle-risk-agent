"""Tests for explicit policy abstention below grounding threshold and fail-loud technical errors."""

# story: e02s03

import hashlib

import pytest

from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyPassage
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter, FakeRerankerAdapter
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex
from vehicle_risk_agent.retrieval.service import (
    HybridRetrievalService,
    PolicyRetrievalError,
)


@pytest.fixture
def sample_passages() -> list[PolicyPassage]:
    return [
        PolicyPassage(
            id="snap1:p001",
            snapshot_id="snap1",
            source_id="nz-legislation-fta-1986",
            section_identifier="Section 9",
            heading="Misleading conduct",
            text="Misleading and deceptive conduct regarding used vehicle sales is prohibited.",
            sequence=1,
            char_offset_start=0,
            char_offset_end=76,
            content_hash=hashlib.sha256(
                b"Misleading and deceptive conduct regarding used vehicle sales is prohibited."
            ).hexdigest(),
        ),
    ]


@pytest.mark.asyncio
async def test_explicit_abstention_when_no_passage_meets_threshold(
    sample_passages: list[PolicyPassage],
) -> None:
    """Queries with no relevant policy support return explicit Abstention."""
    config = RetrievalConfiguration(minimum_reranker_score=0.35)
    embedder = FakeEmbeddingAdapter()
    reranker = FakeRerankerAdapter()
    index = InMemoryPolicyIndex(embedder=embedder, config=config)
    await index.build_index(sample_passages)

    service = HybridRetrievalService(index=index, reranker=reranker, config=config)

    # Completely unrelated query
    result = await service.retrieve(query="astronomy planetary orbital dynamics in deep space")

    assert result.is_abstention is True
    assert len(result.citations) == 0


@pytest.mark.asyncio
async def test_technical_failure_raises_policy_retrieval_error(
    sample_passages: list[PolicyPassage],
) -> None:
    """Technical failures in retrieval fail loudly as PolicyRetrievalError, not false Abstention."""

    class BrokenReranker:
        async def rerank(self, _query: str, _texts: list[str]) -> list[float]:
            raise RuntimeError("Reranker model inference engine failure")

    config = RetrievalConfiguration()
    index = InMemoryPolicyIndex(embedder=FakeEmbeddingAdapter(), config=config)
    await index.build_index(sample_passages)

    service = HybridRetrievalService(index=index, reranker=BrokenReranker(), config=config)

    with pytest.raises(PolicyRetrievalError, match="Technical policy retrieval failure"):
        await service.retrieve(query="misleading vehicle sales")


@pytest.mark.asyncio
async def test_empty_query_returns_explicit_abstention() -> None:
    """Empty or whitespace queries return explicit Abstention."""
    config = RetrievalConfiguration()
    index = InMemoryPolicyIndex(embedder=FakeEmbeddingAdapter(), config=config)
    service = HybridRetrievalService(index=index, reranker=FakeRerankerAdapter(), config=config)

    result = await service.retrieve(query="   \t \n  ")
    assert result.is_abstention is True
    assert len(result.citations) == 0

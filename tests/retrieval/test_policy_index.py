"""Tests for policy passage indexing, dense candidate retrieval, and full-text keyword retrieval."""

import pytest

from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyPassage
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex, RankedCandidate


@pytest.fixture
def sample_passages() -> list[PolicyPassage]:
    return [
        PolicyPassage(
            id="snap1:p001",
            snapshot_id="snap1",
            source_id="nz-legislation-fta-1986",
            section_identifier="Section 9",
            heading="Misleading and deceptive conduct generally",
            text="No person shall, in trade, engage in conduct that is misleading or deceptive.",
            sequence=1,
            char_offset_start=0,
            char_offset_end=78,
            content_hash="a" * 64,
        ),
        PolicyPassage(
            id="snap1:p002",
            snapshot_id="snap1",
            source_id="nz-legislation-fta-1986",
            section_identifier="Section 13",
            heading="False or misleading representations",
            text="No person shall make false representations concerning vehicle history, odometer, or quality.",
            sequence=2,
            char_offset_start=0,
            char_offset_end=92,
            content_hash="b" * 64,
        ),
        PolicyPassage(
            id="snap2:p001",
            snapshot_id="snap2",
            source_id="ppsr-guide",
            section_identifier="Section 1",
            heading="Security Interests on Motor Vehicles",
            text="A registered security interest on the PPSR allows a creditor to repossess the motor vehicle.",
            sequence=1,
            char_offset_start=0,
            char_offset_end=93,
            content_hash="c" * 64,
        ),
    ]


@pytest.mark.asyncio
async def test_dense_candidate_search(sample_passages: list[PolicyPassage]) -> None:
    """Dense search returns ranked candidates ordered by vector similarity."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    index = InMemoryPolicyIndex(embedder=embedder, config=RetrievalConfiguration())
    await index.build_index(sample_passages)

    results = await index.search_dense(query="repossessed motor vehicle security interest", top_k=5)
    assert len(results) > 0
    assert results[0].passage_id == "snap2:p001"
    assert results[0].score > 0.0
    assert isinstance(results[0], RankedCandidate)


@pytest.mark.asyncio
async def test_keyword_candidate_search(sample_passages: list[PolicyPassage]) -> None:
    """Keyword search returns matching candidates with term frequencies."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    index = InMemoryPolicyIndex(embedder=embedder, config=RetrievalConfiguration())
    await index.build_index(sample_passages)

    results = await index.search_keyword(query="misleading deceptive conduct", top_k=5)
    assert len(results) > 0
    assert results[0].passage_id == "snap1:p001"
    assert "Misleading" in results[0].passage.heading


@pytest.mark.asyncio
async def test_index_candidate_bounds(sample_passages: list[PolicyPassage]) -> None:
    """Search limits results to configured top_k cap."""
    embedder = FakeEmbeddingAdapter(dimensions=384)
    index = InMemoryPolicyIndex(embedder=embedder, config=RetrievalConfiguration(dense_candidates=1))
    await index.build_index(sample_passages)

    results = await index.search_dense(query="vehicle", top_k=1)
    assert len(results) == 1

"""Tests for reciprocal rank fusion, cross-encoder reranking, and citation generation."""

import pytest

from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyCitation, PolicyPassage
from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter, FakeRerankerAdapter
from vehicle_risk_agent.retrieval.fusion import reciprocal_rank_fusion
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex, RankedCandidate
from vehicle_risk_agent.retrieval.service import HybridRetrievalService


@pytest.fixture
def sample_passages() -> list[PolicyPassage]:
    return [
        PolicyPassage(
            id="snap1:p001",
            snapshot_id="snap1",
            source_id="nz-legislation-fta-1986",
            section_identifier="Section 9",
            heading="Misleading and deceptive conduct generally",
            text="No person shall engage in misleading or deceptive conduct in vehicle trade.",
            sequence=1,
            char_offset_start=0,
            char_offset_end=74,
            content_hash="a" * 64,
        ),
        PolicyPassage(
            id="snap1:p002",
            snapshot_id="snap1",
            source_id="nz-legislation-fta-1986",
            section_identifier="Section 13",
            heading="False representations about vehicle history",
            text="False representations concerning vehicle odometer are illegal.",
            sequence=2,
            char_offset_start=0,
            char_offset_end=62,
            content_hash="b" * 64,
        ),
        PolicyPassage(
            id="snap2:p001",
            snapshot_id="snap2",
            source_id="ppsr-guide",
            section_identifier="Section 1",
            heading="Security Interests on Motor Vehicles",
            text="A registered security interest allows a creditor to repossess the motor vehicle.",
            sequence=1,
            char_offset_start=0,
            char_offset_end=79,
            content_hash="c" * 64,
        ),
    ]


def test_reciprocal_rank_fusion_logic(sample_passages: list[PolicyPassage]) -> None:
    """RRF combines dense and keyword candidate ranks with k=60."""
    dense = [
        RankedCandidate(passage_id="p1", passage=sample_passages[0], score=0.9, rank=1),
        RankedCandidate(passage_id="p2", passage=sample_passages[1], score=0.8, rank=2),
    ]
    keyword = [
        RankedCandidate(passage_id="p2", passage=sample_passages[1], score=0.95, rank=1),
        RankedCandidate(passage_id="p1", passage=sample_passages[0], score=0.7, rank=2),
    ]

    fused = reciprocal_rank_fusion(
        dense_candidates=dense, keyword_candidates=keyword, rrf_k=60, cap=10
    )

    assert len(fused) == 2
    # p1 score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01613 = 0.03252
    # p2 score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.03252
    assert fused[0].score > 0.03
    assert fused[0].rank == 1


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_stable_citations(
    sample_passages: list[PolicyPassage],
) -> None:
    """Hybrid retrieval fuses, reranks, and returns policy citations."""
    config = RetrievalConfiguration(
        final_passage_cap=5,
        minimum_reranker_score=0.35,
    )
    embedder = FakeEmbeddingAdapter(dimensions=384)
    reranker = FakeRerankerAdapter()
    index = InMemoryPolicyIndex(embedder=embedder, config=config)
    await index.build_index(sample_passages)

    source_metadata = {
        "nz-legislation-fta-1986": ("Fair Trading Act 1986", "https://legislation.govt.nz/fta"),
        "ppsr-guide": ("PPSR Vehicle Guide", "https://ppsr.govt.nz/guide"),
    }

    service = HybridRetrievalService(
        index=index,
        reranker=reranker,
        config=config,
        source_metadata=source_metadata,
    )

    result = await service.retrieve(
        query="What happens if a vehicle has an outstanding security interest?"
    )

    assert not result.is_abstention
    assert len(result.citations) >= 1
    assert result.citations[0].section_identifier == "Section 1"
    assert result.citations[0].source_title == "PPSR Vehicle Guide"
    assert result.citations[0].canonical_origin == "https://ppsr.govt.nz/guide"
    assert isinstance(result.citations[0], PolicyCitation)


@pytest.mark.asyncio
async def test_hybrid_retrieval_caps_at_five_passages() -> None:
    """Hybrid retrieval caps final returned citations to at most final_passage_cap (5)."""
    # Create 8 passages
    extra_passages = [
        PolicyPassage(
            id=f"snap1:p{i:03d}",
            snapshot_id="snap1",
            source_id="nz-legislation-fta-1986",
            section_identifier=f"Section {i}",
            heading=f"Heading {i}",
            text=f"Content for vehicle trade clause {i}.",
            sequence=i,
            char_offset_start=0,
            char_offset_end=35,
            content_hash=f"{i:02d}" * 32,
        )
        for i in range(1, 9)
    ]
    config = RetrievalConfiguration(final_passage_cap=5, minimum_reranker_score=0.1)
    index = InMemoryPolicyIndex(embedder=FakeEmbeddingAdapter(), config=config)
    await index.build_index(extra_passages)

    service = HybridRetrievalService(
        index=index,
        reranker=FakeRerankerAdapter(),
        config=config,
        source_metadata={"nz-legislation-fta-1986": ("FTA", "https://example.com")},
    )

    result = await service.retrieve(query="vehicle trade")
    assert len(result.citations) <= 5

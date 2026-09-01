"""Policy retrieval, ranking, fusion, and citation grounding package."""

from vehicle_risk_agent.retrieval.adapters import (
    EmbeddingAdapter,
    FakeEmbeddingAdapter,
    FakeRerankerAdapter,
    RerankerAdapter,
)
from vehicle_risk_agent.retrieval.fusion import reciprocal_rank_fusion
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex, RankedCandidate
from vehicle_risk_agent.retrieval.service import HybridRetrievalService, RetrievalResult

__all__ = [
    "EmbeddingAdapter",
    "FakeEmbeddingAdapter",
    "FakeRerankerAdapter",
    "HybridRetrievalService",
    "InMemoryPolicyIndex",
    "RankedCandidate",
    "RerankerAdapter",
    "RetrievalResult",
    "reciprocal_rank_fusion",
]

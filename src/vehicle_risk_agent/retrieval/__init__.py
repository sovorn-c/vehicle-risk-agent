"""Policy retrieval, ranking, fusion, and citation grounding package."""

from vehicle_risk_agent.retrieval.adapters import (
    CrossEncoderRerankerAdapter,
    EmbeddingAdapter,
    FakeEmbeddingAdapter,
    FakeRerankerAdapter,
    RerankerAdapter,
    SentenceTransformersEmbeddingAdapter,
)
from vehicle_risk_agent.retrieval.fusion import reciprocal_rank_fusion
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex, PolicyIndex, RankedCandidate
from vehicle_risk_agent.retrieval.postgres_index import PostgresPolicyIndex
from vehicle_risk_agent.retrieval.service import (
    HybridRetrievalService,
    PolicyRetrievalError,
    RetrievalResult,
)

__all__ = [
    "CrossEncoderRerankerAdapter",
    "EmbeddingAdapter",
    "FakeEmbeddingAdapter",
    "FakeRerankerAdapter",
    "HybridRetrievalService",
    "InMemoryPolicyIndex",
    "PolicyIndex",
    "PolicyRetrievalError",
    "PostgresPolicyIndex",
    "RankedCandidate",
    "RerankerAdapter",
    "RetrievalResult",
    "SentenceTransformersEmbeddingAdapter",
    "reciprocal_rank_fusion",
]

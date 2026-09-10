"""Domain models for Policy Corpus Manifests and Retrieval Configuration."""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CorpusLifecycleState(StrEnum):
    """Lifecycle state machine for Policy Corpus Versions."""

    DRAFT = "DRAFT"
    READY = "READY"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class RetrievalConfiguration(BaseModel):
    """Pinned retrieval, dense search, full-text search, fusion, and reranking parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: str = "neural"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "main"
    embedding_dimensions: int = Field(default=384, gt=0)
    normalize_embeddings: bool = True
    dense_candidates: int = Field(default=20, gt=0)
    keyword_candidates: int = Field(default=20, gt=0)
    fusion: str = "reciprocal-rank-fusion"
    rrf_k: int = Field(default=60, gt=0)
    fused_candidate_cap: int = Field(default=40, gt=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_revision: str = "main"
    reranker_activation: str = "sigmoid"
    rerank_candidate_cap: int = Field(default=20, gt=0)
    final_passage_cap: int = Field(default=5, gt=0)
    minimum_reranker_score: float = Field(default=0.35, ge=0.0, le=1.0)


def compute_manifest_hash(
    corpus_id: str,
    snapshot_ids: list[str] | tuple[str, ...],
    retrieval_config: RetrievalConfiguration,
) -> str:
    """Compute deterministic SHA-256 hash of corpus manifest contents."""
    canonical = {
        "corpus_id": corpus_id,
        "snapshot_ids": list(snapshot_ids),
        "retrieval_config": retrieval_config.model_dump(mode="json"),
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


class PolicyCorpusManifest(BaseModel):
    """Immutable versioned manifest pinning active Policy Snapshots and retrieval settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=1024)
    lifecycle_state: CorpusLifecycleState = CorpusLifecycleState.DRAFT
    snapshot_ids: tuple[str, ...] = Field(min_length=1)
    retrieval_config: RetrievalConfiguration = Field(default_factory=RetrievalConfiguration)
    manifest_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    activated_at: datetime | None = None
    retired_at: datetime | None = None

    @field_validator("snapshot_ids", mode="before")
    @classmethod
    def validate_unique_snapshots(cls, v: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Ensure snapshot_ids list has no duplicates and contains non-empty strings."""
        tuple_val = tuple(v)
        if not tuple_val:
            raise ValueError("snapshot_ids cannot be empty")
        if len(tuple_val) != len(set(tuple_val)):
            raise ValueError("snapshot_ids cannot contain duplicates")
        for s in tuple_val:
            if not s or not s.strip():
                raise ValueError("snapshot_id cannot be empty")
        return tuple_val


def build_corpus_manifest(
    corpus_id: str,
    name: str,
    description: str,
    snapshot_ids: list[str] | tuple[str, ...],
    retrieval_config: RetrievalConfiguration | None = None,
    lifecycle_state: CorpusLifecycleState = CorpusLifecycleState.DRAFT,
) -> PolicyCorpusManifest:
    """Construct a PolicyCorpusManifest with computed manifest hash."""
    if retrieval_config is None:
        retrieval_config = RetrievalConfiguration()

    manifest_hash = compute_manifest_hash(corpus_id, snapshot_ids, retrieval_config)

    return PolicyCorpusManifest(
        id=corpus_id,
        name=name,
        description=description,
        lifecycle_state=lifecycle_state,
        snapshot_ids=tuple(snapshot_ids),
        retrieval_config=retrieval_config,
        manifest_hash=manifest_hash,
    )

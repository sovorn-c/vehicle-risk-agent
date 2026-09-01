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

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    dense_candidates: int = 20
    keyword_candidates: int = 20
    fusion: str = "reciprocal-rank-fusion"
    rrf_k: int = 60
    fused_candidate_cap: int = 40
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_activation: str = "sigmoid"
    rerank_candidate_cap: int = 20
    final_passage_cap: int = 5
    minimum_reranker_score: float = 0.35


def compute_manifest_hash(
    corpus_id: str,
    snapshot_ids: list[str],
    retrieval_config: RetrievalConfiguration,
) -> str:
    """Compute deterministic SHA-256 hash of corpus manifest contents."""
    canonical = {
        "corpus_id": corpus_id,
        "snapshot_ids": snapshot_ids,
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
    snapshot_ids: list[str] = Field(min_length=1)
    retrieval_config: RetrievalConfiguration = Field(default_factory=RetrievalConfiguration)
    manifest_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    activated_at: datetime | None = None
    retired_at: datetime | None = None

    @field_validator("snapshot_ids")
    @classmethod
    def validate_unique_snapshots(cls, v: list[str]) -> list[str]:
        """Ensure snapshot_ids list has no duplicates and contains non-empty strings."""
        if not v:
            raise ValueError("snapshot_ids cannot be empty")
        if len(v) != len(set(v)):
            raise ValueError("snapshot_ids cannot contain duplicates")
        for s in v:
            if not s or not s.strip():
                raise ValueError("snapshot_id cannot be empty")
        return v


def build_corpus_manifest(
    corpus_id: str,
    name: str,
    description: str,
    snapshot_ids: list[str],
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
        snapshot_ids=snapshot_ids,
        retrieval_config=retrieval_config,
        manifest_hash=manifest_hash,
    )

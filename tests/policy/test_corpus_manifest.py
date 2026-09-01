"""Tests for Policy Corpus Manifest creation, versioning, immutability, and hash pinning."""

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.policy.corpus_models import (
    CorpusLifecycleState,
    RetrievalConfiguration,
    build_corpus_manifest,
)


def test_retrieval_configuration_defaults() -> None:
    """RetrievalConfiguration defaults match GATE-02 specification."""
    config = RetrievalConfiguration()

    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.embedding_dimensions == 384
    assert config.dense_candidates == 20
    assert config.keyword_candidates == 20
    assert config.fusion == "reciprocal-rank-fusion"
    assert config.rrf_k == 60
    assert config.fused_candidate_cap == 40
    assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.reranker_activation == "sigmoid"
    assert config.rerank_candidate_cap == 20
    assert config.final_passage_cap == 5
    assert config.minimum_reranker_score == 0.35


def test_corpus_manifest_creation_and_hash() -> None:
    """build_corpus_manifest produces a deterministic manifest hash."""
    snapshot_ids = ["nz-fta-1986:snap1", "ppsr-guide:snap1"]
    manifest = build_corpus_manifest(
        corpus_id="corpus-v1",
        name="NZ Motor Vehicle Policy Corpus",
        description="Official NZ consumer protection and PPSR guidance.",
        snapshot_ids=snapshot_ids,
        retrieval_config=RetrievalConfiguration(),
    )

    assert manifest.id == "corpus-v1"
    assert manifest.lifecycle_state == CorpusLifecycleState.DRAFT
    assert manifest.snapshot_ids == tuple(snapshot_ids)
    assert manifest.manifest_hash is not None
    assert len(manifest.manifest_hash) == 64

    # Identical inputs produce identical hash
    manifest_dup = build_corpus_manifest(
        corpus_id="corpus-v1",
        name="NZ Motor Vehicle Policy Corpus",
        description="Official NZ consumer protection and PPSR guidance.",
        snapshot_ids=snapshot_ids,
        retrieval_config=RetrievalConfiguration(),
    )
    assert manifest_dup.manifest_hash == manifest.manifest_hash


def test_corpus_manifest_rejects_empty_or_duplicate_snapshots_on_validation() -> None:
    """Manifest rejects empty snapshot list or duplicate snapshots."""
    with pytest.raises(ValidationError):
        build_corpus_manifest(
            corpus_id="corpus-v1",
            name="Empty Corpus",
            description="No snapshots",
            snapshot_ids=[],  # Empty forbidden
            retrieval_config=RetrievalConfiguration(),
        )

    with pytest.raises(ValidationError):
        build_corpus_manifest(
            corpus_id="corpus-v1",
            name="Duplicate Corpus",
            description="Duplicate snapshots",
            snapshot_ids=["snap1", "snap1"],  # Duplicate forbidden
            retrieval_config=RetrievalConfiguration(),
        )


def test_corpus_manifest_immutability() -> None:
    """PolicyCorpusManifest instances are frozen models."""
    manifest = build_corpus_manifest(
        corpus_id="corpus-v1",
        name="NZ Corpus",
        description="Desc",
        snapshot_ids=["snap1"],
        retrieval_config=RetrievalConfiguration(),
    )

    attr_name = "name"
    with pytest.raises(ValidationError):
        setattr(manifest, attr_name, "Modified Name")


def test_corpus_manifest_snapshot_ids_is_immutable_tuple() -> None:
    """PolicyCorpusManifest snapshot_ids is an immutable sequence/tuple."""
    manifest = build_corpus_manifest(
        corpus_id="corpus-v1",
        name="NZ Corpus",
        description="Desc",
        snapshot_ids=["snap1", "snap2"],
        retrieval_config=RetrievalConfiguration(),
    )

    assert isinstance(manifest.snapshot_ids, tuple)
    with pytest.raises(AttributeError):
        manifest.snapshot_ids.append("snap3")  # type: ignore[attr-defined]


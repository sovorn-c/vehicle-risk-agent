"""Pinned, attributable corpus fixture for the E12 retrieval benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vehicle_risk_agent.policy.corpus_models import (
    PolicyCorpusManifest,
    RetrievalConfiguration,
    build_corpus_manifest,
)
from vehicle_risk_agent.policy.ingestion import PolicyParser, ingest_policy_source
from vehicle_risk_agent.policy.models import (
    AuthorityClassification,
    PolicyPassage,
    PolicySnapshot,
    PolicySource,
)


class _SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    issuing_authority: str = Field(min_length=1)
    canonical_origin: str = Field(min_length=1)
    authority_classification: AuthorityClassification
    reuse_terms: str = Field(min_length=1)
    expected_update_cadence: str = Field(min_length=1)
    source_version_as_at: date
    publication_date: datetime | None = None
    effective_date: datetime | None = None
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: str = Field(min_length=1)


class _CorpusRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    corpus_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    retrieved_at: datetime
    retrieval_config: RetrievalConfiguration
    sources: tuple[_SourceRecord, ...] = Field(min_length=4, max_length=8)


@dataclass(frozen=True)
class PortfolioCorpusBundle:
    """Validated domain objects constructed from the checked-in corpus manifest."""

    manifest: PolicyCorpusManifest
    sources: tuple[PolicySource, ...]
    snapshots: tuple[PolicySnapshot, ...]

    @property
    def snapshot_ids(self) -> tuple[str, ...]:
        return self.manifest.snapshot_ids

    @property
    def passages(self) -> tuple[PolicyPassage, ...]:
        return tuple(passage for snapshot in self.snapshots for passage in snapshot.passages)

    @property
    def fingerprint(self) -> str:
        return self.manifest.manifest_hash


def load_portfolio_corpus(manifest_path: Path) -> PortfolioCorpusBundle:
    """Load and validate source attribution, reuse terms, hashes, and retrieval settings."""
    try:
        raw_record = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = _CorpusRecord.model_validate(raw_record)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid portfolio corpus manifest: {exc}") from exc

    if record.schema_version != 1:
        raise ValueError(f"Unsupported portfolio corpus schema version: {record.schema_version}")

    sources: list[PolicySource] = []
    snapshots: list[PolicySnapshot] = []
    source_ids: set[str] = set()
    for entry in record.sources:
        if entry.id in source_ids:
            raise ValueError(f"Duplicate portfolio corpus source id: {entry.id}")
        source_ids.add(entry.id)
        if not entry.canonical_origin.startswith("https://www.legislation.govt.nz/"):
            raise ValueError(
                f"Portfolio corpus source is not an official legislation URL: {entry.id}"
            )

        content_hash = hashlib.sha256(entry.content.encode("utf-8")).hexdigest()
        expected_hash = entry.content_sha256
        if expected_hash != content_hash:
            raise ValueError(f"Portfolio corpus source hash mismatch: {entry.id}")

        source = PolicySource(
            id=entry.id,
            title=entry.title,
            issuing_authority=entry.issuing_authority,
            canonical_origin=entry.canonical_origin,
            authority_classification=entry.authority_classification,
            reuse_terms=entry.reuse_terms,
            expected_update_cadence=entry.expected_update_cadence,
        )
        snapshot = ingest_policy_source(
            source=source,
            raw_content=entry.content,
            parser=PolicyParser(),
            effective_date=entry.effective_date,
            publication_date=entry.publication_date,
            metadata={
                "portfolio_corpus": record.corpus_id,
                "source_retrieved_at": record.retrieved_at.isoformat(),
                "source_version_as_at": entry.source_version_as_at.isoformat(),
                "source_content_sha256": content_hash,
                "canonical_origin": entry.canonical_origin,
                "reuse_terms": entry.reuse_terms,
            },
        )
        sources.append(source)
        snapshots.append(snapshot)

    corpus_manifest = build_corpus_manifest(
        corpus_id=record.corpus_id,
        name=record.name,
        description=record.description,
        snapshot_ids=[snapshot.id for snapshot in snapshots],
        retrieval_config=record.retrieval_config,
    )
    return PortfolioCorpusBundle(
        manifest=corpus_manifest,
        sources=tuple(sources),
        snapshots=tuple(snapshots),
    )

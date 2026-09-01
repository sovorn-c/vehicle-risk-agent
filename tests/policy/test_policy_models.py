"""Tests for Policy Source, Snapshot, Passage, and metadata domain models."""

import hashlib
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.policy.models import (
    AuthorityClassification,
    PolicyPassage,
    PolicySnapshot,
    PolicySource,
    SourceStatus,
    ValidationOutcome,
)


def test_policy_source_creation_and_immutability() -> None:
    """A valid PolicySource can be created and is immutable."""
    source = PolicySource(
        id="nz-legislation-fta-1986",
        title="Fair Trading Act 1986",
        issuing_authority="Parliament of New Zealand",
        jurisdiction="NZ",
        canonical_origin="https://www.legislation.govt.nz/act/public/1986/0121/latest/DLM96439.html",
        authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
        reuse_terms="Crown copyright (CC BY 4.0)",
        expected_update_cadence="ADHOC",
        status=SourceStatus.ACTIVE,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert source.id == "nz-legislation-fta-1986"
    assert source.authority_classification == AuthorityClassification.PRIMARY_LEGISLATION
    assert source.jurisdiction == "NZ"

    # Frozen/immutable check
    with pytest.raises(ValidationError):
        source.title = "Changed Title"  # type: ignore[misc]


def test_policy_source_rejects_extra_and_invalid_fields() -> None:
    """PolicySource rejects unknown fields, empty fields, and invalid jurisdiction."""
    with pytest.raises(ValidationError):
        PolicySource(
            id="nz-fta",
            title="",  # Empty title
            issuing_authority="NZ Govt",
            jurisdiction="NZ",
            canonical_origin="https://example.com",
            authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
            reuse_terms="CC BY 4.0",
            expected_update_cadence="ADHOC",
        )

    with pytest.raises(ValidationError):
        PolicySource(
            id="nz-fta",
            title="Fair Trading Act",
            issuing_authority="NZ Govt",
            jurisdiction="NZ",
            canonical_origin="https://example.com",
            authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
            reuse_terms="CC BY 4.0",
            expected_update_cadence="ADHOC",
            unknown_field="injected",  # Extra field forbidden
        )  # type: ignore[call-arg]


def test_policy_passage_attributes_and_bounds() -> None:
    """PolicyPassage captures section identity, heading, sequence, text, and hash."""
    text = "Section 9: No person shall, in trade, engage in conduct that is misleading or deceptive."
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    passage = PolicyPassage(
        id="nz-legislation-fta-1986:snap1:p001",
        snapshot_id="nz-legislation-fta-1986:snap1",
        source_id="nz-legislation-fta-1986",
        section_identifier="Section 9",
        heading="Misleading and deceptive conduct generally",
        text=text,
        sequence=1,
        char_offset_start=0,
        char_offset_end=len(text),
        content_hash=content_hash,
    )

    assert passage.section_identifier == "Section 9"
    assert passage.sequence == 1
    assert passage.content_hash == content_hash
    assert len(passage.text) <= 1200


def test_policy_passage_rejects_empty_or_oversize_text() -> None:
    """PolicyPassage rejects empty text or text exceeding chunk limit (1200 chars)."""
    with pytest.raises(ValidationError):
        PolicyPassage(
            id="p1",
            snapshot_id="s1",
            source_id="src1",
            section_identifier="Sec 1",
            heading="Head",
            text="",  # Empty
            sequence=1,
            char_offset_start=0,
            char_offset_end=0,
            content_hash=hashlib.sha256(b"").hexdigest(),
        )

    oversize_text = "A" * 1500
    with pytest.raises(ValidationError):
        PolicyPassage(
            id="p1",
            snapshot_id="s1",
            source_id="src1",
            section_identifier="Sec 1",
            heading="Head",
            text=oversize_text,  # Exceeds max 1200
            sequence=1,
            char_offset_start=0,
            char_offset_end=1500,
            content_hash=hashlib.sha256(oversize_text.encode("utf-8")).hexdigest(),
        )


def test_policy_snapshot_valid() -> None:
    """PolicySnapshot binds source_id, raw_content, content_hash, and passages."""
    raw_content = "# Fair Trading Act 1986\n## Section 9\nNo person shall in trade..."
    content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    now = datetime.now(UTC)

    snapshot = PolicySnapshot(
        id=f"nz-legislation-fta-1986:{content_hash[:12]}",
        source_id="nz-legislation-fta-1986",
        retrieved_at=now,
        effective_date=datetime(2026, 1, 1, tzinfo=UTC),
        publication_date=datetime(1986, 12, 17, tzinfo=UTC),
        content_hash=content_hash,
        raw_content=raw_content,
        parser_version="policy-parser-v1",
        validation_outcome=ValidationOutcome.VALID,
        passages=[],
        metadata={"source_title": "Fair Trading Act 1986"},
    )

    assert snapshot.source_id == "nz-legislation-fta-1986"
    assert snapshot.content_hash == content_hash
    assert snapshot.validation_outcome == ValidationOutcome.VALID


def test_policy_snapshot_rejects_hash_mismatch() -> None:
    """PolicySnapshot rejects mismatched content_hash."""
    raw_content = "Real content"
    wrong_hash = hashlib.sha256(b"Different content").hexdigest()

    with pytest.raises(ValidationError, match="content_hash does not match"):
        PolicySnapshot(
            id="snap1",
            source_id="src1",
            retrieved_at=datetime.now(UTC),
            content_hash=wrong_hash,
            raw_content=raw_content,
            parser_version="policy-parser-v1",
            validation_outcome=ValidationOutcome.VALID,
            passages=[],
        )

"""Tests for policy source parsing, chunking, structural validation, and ingestion."""

import hashlib
from datetime import UTC, datetime

import pytest

from vehicle_risk_agent.policy.ingestion import (
    PolicyIngestionError,
    PolicyParser,
    ingest_policy_source,
)
from vehicle_risk_agent.policy.models import (
    AuthorityClassification,
    PolicySource,
    SourceStatus,
    ValidationOutcome,
)


@pytest.fixture
def sample_fta_source() -> PolicySource:
    return PolicySource(
        id="nz-legislation-fta-1986",
        title="Fair Trading Act 1986",
        issuing_authority="Parliament of New Zealand",
        jurisdiction="NZ",
        canonical_origin="https://www.legislation.govt.nz/act/public/1986/0121/latest/DLM96439.html",
        authority_classification=AuthorityClassification.PRIMARY_LEGISLATION,
        reuse_terms="Crown copyright (CC BY 4.0)",
        expected_update_cadence="ADHOC",
        status=SourceStatus.ACTIVE,
    )


@pytest.fixture
def sample_ppsr_source() -> PolicySource:
    return PolicySource(
        id="ppsr-guidance-buying-used-car",
        title="PPSR Guide: Checking Motor Vehicles for Security Interests",
        issuing_authority="New Zealand Companies Office",
        jurisdiction="NZ",
        canonical_origin="https://www.ppsr.govt.nz/buyer-guide-vehicles",
        authority_classification=AuthorityClassification.OFFICIAL_GUIDANCE,
        reuse_terms="Crown copyright (CC BY 4.0)",
        expected_update_cadence="ANNUAL",
        status=SourceStatus.ACTIVE,
    )


def test_parser_extracts_sections_and_passages(sample_fta_source: PolicySource) -> None:
    """Parser parses markdown sections into attributable passages."""
    raw_markdown = """# Fair Trading Act 1986

## Section 9: Misleading and deceptive conduct generally
No person shall, in trade, engage in conduct that is misleading or deceptive.

## Section 13: False or misleading representations
No person shall, in trade, make a false or misleading representation that goods
are of a particular kind, standard, quality, grade, or have had a particular history.
"""

    snapshot = ingest_policy_source(
        source=sample_fta_source,
        raw_content=raw_markdown,
        parser=PolicyParser(),
        effective_date=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert snapshot.source_id == sample_fta_source.id
    assert snapshot.validation_outcome == ValidationOutcome.VALID
    assert snapshot.content_hash == hashlib.sha256(raw_markdown.encode("utf-8")).hexdigest()
    assert len(snapshot.passages) == 2

    p1 = snapshot.passages[0]
    assert p1.section_identifier == "Section 9"
    assert "Misleading and deceptive conduct" in p1.heading
    assert "No person shall, in trade" in p1.text
    assert p1.sequence == 1
    assert p1.source_id == sample_fta_source.id
    assert p1.snapshot_id == snapshot.id

    p2 = snapshot.passages[1]
    assert p2.section_identifier == "Section 13"
    assert "False or misleading representations" in p2.heading
    assert p2.sequence == 2


def test_parser_splits_oversize_section_at_paragraphs(sample_ppsr_source: PolicySource) -> None:
    """Sections over 1200 characters are split at paragraph boundaries with bounded overlap."""
    paragraph_1 = "Paragraph 1: " + (
        "A search on the PPSR shows if a motor vehicle has money owing on it. " * 15
    )
    paragraph_2 = "Paragraph 2: " + (
        "If you buy a vehicle with a registered security interest, it could be repossessed. " * 15
    )
    paragraph_3 = "Paragraph 3: " + (
        "Always search by VIN and chassis number before completing any vehicle purchase. " * 15
    )

    raw_markdown = f"""# PPSR Buyer Guide

## Section 1: Overview and Repossession Risk
{paragraph_1}

{paragraph_2}

{paragraph_3}
"""

    snapshot = ingest_policy_source(
        source=sample_ppsr_source,
        raw_content=raw_markdown,
        parser=PolicyParser(max_passage_chars=1200, overlap_chars=150),
    )

    assert snapshot.validation_outcome == ValidationOutcome.VALID
    assert len(snapshot.passages) >= 2
    for p in snapshot.passages:
        assert len(p.text) <= 1200
        assert p.section_identifier == "Section 1"
        assert p.content_hash == hashlib.sha256(p.text.encode("utf-8")).hexdigest()


def test_parser_rejects_empty_content(sample_fta_source: PolicySource) -> None:
    """Ingestion rejects empty raw content."""
    with pytest.raises(PolicyIngestionError, match="Content cannot be empty"):
        ingest_policy_source(
            source=sample_fta_source,
            raw_content="   \n  \t  ",
            parser=PolicyParser(),
        )


def test_parser_rejects_oversize_content(sample_fta_source: PolicySource) -> None:
    """Ingestion rejects raw content exceeding maximum document size (5MB)."""
    oversize = "A" * (5 * 1024 * 1024 + 10)
    with pytest.raises(PolicyIngestionError, match="exceeds maximum allowed size"):
        ingest_policy_source(
            source=sample_fta_source,
            raw_content=oversize,
            parser=PolicyParser(),
        )


def test_parser_rejects_null_bytes_and_unsafe_control_chars(
    sample_fta_source: PolicySource,
) -> None:
    """Ingestion rejects content with null bytes or unsafe control characters."""
    with pytest.raises(PolicyIngestionError, match="unsafe control characters"):
        ingest_policy_source(
            source=sample_fta_source,
            raw_content="# Heading\nSection \x00 with null byte",
            parser=PolicyParser(),
        )


def test_parser_deterministic_reproducibility(sample_fta_source: PolicySource) -> None:
    """Parsing identical content produces identical snapshot ID and passage IDs."""
    raw_markdown = """# NZTA VIRM
## Section 1-1: Structure and Corrosion
Vehicle structure must be free from structural corrosion and damage.
"""
    s1 = ingest_policy_source(sample_fta_source, raw_markdown, PolicyParser())
    s2 = ingest_policy_source(sample_fta_source, raw_markdown, PolicyParser())

    assert s1.id == s2.id
    assert s1.content_hash == s2.content_hash
    assert [p.id for p in s1.passages] == [p.id for p in s2.passages]


def test_exact_character_offsets_and_lossless_overlap(sample_ppsr_source: PolicySource) -> None:
    """Passages report exact character offsets and bounded overlap without text loss."""
    p1 = "Paragraph 1: " + ("First section text detailing PPSR registration rules. " * 20)
    p2 = "Paragraph 2: " + ("Second section text explaining repossession priorities. " * 20)
    raw_markdown = f"# Guide\n\n## Section 1: PPSR Rules\n{p1}\n\n{p2}\n"

    snapshot = ingest_policy_source(
        source=sample_ppsr_source,
        raw_content=raw_markdown,
        parser=PolicyParser(max_passage_chars=800, overlap_chars=100),
    )

    assert len(snapshot.passages) >= 2
    for p in snapshot.passages:
        # Offsets must be strictly within document bounds
        assert 0 <= p.char_offset_start < p.char_offset_end <= len(raw_markdown)
        # Slicing raw_content with offsets must match passage text exactly
        assert raw_markdown[p.char_offset_start : p.char_offset_end] == p.text
        assert len(p.text) <= 800

    # Test overlap between consecutive passages in the same section
    for i in range(len(snapshot.passages) - 1):
        curr_p = snapshot.passages[i]
        next_p = snapshot.passages[i + 1]
        assert curr_p.char_offset_end > next_p.char_offset_start
        overlap_len = curr_p.char_offset_end - next_p.char_offset_start
        assert 0 < overlap_len <= 150

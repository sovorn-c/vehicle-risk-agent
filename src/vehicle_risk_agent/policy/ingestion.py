"""Policy document parsing, chunking, structural validation, and ingestion."""

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from vehicle_risk_agent.policy.models import (
    PolicyPassage,
    PolicySnapshot,
    PolicySource,
    ValidationOutcome,
)

MAX_DOCUMENT_SIZE_BYTES = 5 * 1024 * 1024  # 5MB
DEFAULT_MAX_PASSAGE_CHARS = 1200  # GATE-02
DEFAULT_OVERLAP_CHARS = 150  # GATE-02
DEFAULT_PARSER_VERSION = "policy-parser-v1"


class PolicyIngestionError(Exception):
    """Raised when policy document content is invalid, unsafe, or fails parsing."""


def _check_unsafe_characters(content: str) -> None:
    """Check for null bytes and prohibited control characters."""
    for char in content:
        code = ord(char)
        if code == 0 or (code < 32 and char not in ("\n", "\r", "\t")):
            raise PolicyIngestionError("Content contains unsafe control characters")


def _extract_section_identifier(heading: str) -> str:
    """Extract canonical section identifier from a heading if present."""
    match = re.search(
        r"((?:Section|Part|Clause|Rule|Schedule|Schedule\s+\d+|App)\s+[\w\d\-\.]+)",
        heading,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    clean = re.sub(r"^[#\s]+", "", heading).strip()
    return clean[:64] if clean else "General"


class PolicyParser:
    """Parses markdown policy documents into structured, bounded, attributable passages."""

    def __init__(
        self,
        max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        version: str = DEFAULT_PARSER_VERSION,
    ) -> None:
        self.max_passage_chars = max_passage_chars
        self.overlap_chars = overlap_chars
        self.version = version

    def parse(
        self,
        raw_content: str,
        snapshot_id: str,
        source_id: str,
    ) -> list[PolicyPassage]:
        """Parse raw content into section-level passages with exact offsets and bounded overlap."""
        # Find all markdown headings (## or ###) with their character positions
        heading_matches = list(re.finditer(r"^(#{2,3})\s+(.+)$", raw_content, re.MULTILINE))

        section_spans: list[tuple[str, str, int, int]] = []  # (sec_id, heading, start_idx, end_idx)

        if not heading_matches:
            doc_heading = "Overview"
            first_h1 = re.search(r"^#\s+(.+)$", raw_content, re.MULTILINE)
            if first_h1:
                doc_heading = first_h1.group(1).strip()
            section_spans.append(("Overview", doc_heading, 0, len(raw_content)))
        else:
            # Attach the document prefix to the first section so every source
            # character remains attributable without creating a title-only chunk.
            for idx, match in enumerate(heading_matches):
                heading_text = match.group(2).strip()
                sec_id = _extract_section_identifier(heading_text)
                start_pos = 0 if idx == 0 else match.start()
                end_pos = (
                    heading_matches[idx + 1].start()
                    if idx + 1 < len(heading_matches)
                    else len(raw_content)
                )
                section_spans.append((sec_id, heading_text, start_pos, end_pos))

        passages: list[PolicyPassage] = []
        sequence = 1

        for sec_id, heading, sec_start, sec_end in section_spans:
            sec_len = sec_end - sec_start
            if sec_len <= self.max_passage_chars:
                passage_text = raw_content[sec_start:sec_end]
                passage_hash = hashlib.sha256(passage_text.encode("utf-8")).hexdigest()
                passages.append(
                    PolicyPassage(
                        id=f"{snapshot_id}:p{sequence:03d}",
                        snapshot_id=snapshot_id,
                        source_id=source_id,
                        section_identifier=sec_id,
                        heading=heading,
                        text=passage_text,
                        sequence=sequence,
                        char_offset_start=sec_start,
                        char_offset_end=sec_end,
                        content_hash=passage_hash,
                    )
                )
                sequence += 1
            else:
                curr_rel = 0
                while curr_rel < sec_len:
                    remaining = sec_len - curr_rel
                    if remaining <= self.max_passage_chars:
                        chunk_rel_end = sec_len
                        next_rel = sec_len
                    else:
                        target_end = curr_rel + self.max_passage_chars
                        window_start = max(curr_rel + 1, target_end - self.overlap_chars)
                        window_text = raw_content[sec_start + window_start : sec_start + target_end]

                        best_cut_offset: int | None = None
                        for delim in ("\n\n", "\n", ". ", "? ", "! ", " "):
                            r_idx = window_text.rfind(delim)
                            if r_idx != -1:
                                best_cut_offset = window_start + r_idx + len(delim)
                                break

                        if best_cut_offset is not None and best_cut_offset > curr_rel:
                            chunk_rel_end = best_cut_offset
                        else:
                            chunk_rel_end = target_end

                        next_rel = max(curr_rel + 1, chunk_rel_end - self.overlap_chars)

                    chunk_abs_start = sec_start + curr_rel
                    chunk_abs_end = sec_start + chunk_rel_end
                    chunk_text = raw_content[chunk_abs_start:chunk_abs_end]
                    chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

                    passages.append(
                        PolicyPassage(
                            id=f"{snapshot_id}:p{sequence:03d}",
                            snapshot_id=snapshot_id,
                            source_id=source_id,
                            section_identifier=sec_id,
                            heading=heading,
                            text=chunk_text,
                            sequence=sequence,
                            char_offset_start=chunk_abs_start,
                            char_offset_end=chunk_abs_end,
                            content_hash=chunk_hash,
                        )
                    )
                    sequence += 1
                    curr_rel = next_rel

        return passages


def ingest_policy_source(
    source: PolicySource,
    raw_content: str,
    parser: PolicyParser | None = None,
    effective_date: datetime | None = None,
    publication_date: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> PolicySnapshot:
    """Validate, parse, and content-hash an official policy document into a PolicySnapshot."""
    if parser is None:
        parser = PolicyParser()

    if not raw_content or not raw_content.strip():
        raise PolicyIngestionError("Content cannot be empty")

    content_bytes = raw_content.encode("utf-8")
    if len(content_bytes) > MAX_DOCUMENT_SIZE_BYTES:
        raise PolicyIngestionError(
            f"Content exceeds maximum allowed size ({MAX_DOCUMENT_SIZE_BYTES} bytes)"
        )

    _check_unsafe_characters(raw_content)

    content_hash = hashlib.sha256(content_bytes).hexdigest()
    snapshot_id = f"{source.id}:{content_hash[:16]}"

    passages = parser.parse(
        raw_content=raw_content,
        snapshot_id=snapshot_id,
        source_id=source.id,
    )

    snapshot_metadata = dict(metadata or {})
    snapshot_metadata["source_title"] = source.title
    snapshot_metadata["authority"] = source.issuing_authority
    snapshot_metadata["authority_classification"] = str(source.authority_classification)

    return PolicySnapshot(
        id=snapshot_id,
        source_id=source.id,
        retrieved_at=datetime.now(UTC),
        effective_date=effective_date,
        publication_date=publication_date,
        content_hash=content_hash,
        raw_content=raw_content,
        parser_version=parser.version,
        validation_outcome=ValidationOutcome.VALID,
        passages=tuple(passages),
        metadata=snapshot_metadata,
    )

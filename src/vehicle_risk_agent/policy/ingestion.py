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
    match = re.search(r"((?:Section|Part|Clause|Rule|Schedule|Schedule\s+\d+|App)\s+[\w\d\-\.]+)", heading, re.IGNORECASE)
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
        """Parse raw content into section-level passages, splitting oversized sections."""
        lines = raw_content.splitlines()
        sections: list[tuple[str, str, str]] = []  # (section_id, heading, text)

        current_heading = "Preamble"
        current_section_id = "Overview"
        current_lines: list[str] = []

        for line in lines:
            if line.startswith("## ") or line.startswith("### "):
                if current_lines:
                    text = "\n".join(current_lines).strip()
                    if text:
                        sections.append((current_section_id, current_heading, text))
                    current_lines = []
                heading_text = line.lstrip("#").strip()
                current_heading = heading_text
                current_section_id = _extract_section_identifier(heading_text)
            elif line.startswith("# ") and not current_lines:
                # Document title heading
                heading_text = line.lstrip("#").strip()
                current_heading = heading_text
            else:
                current_lines.append(line)

        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append((current_section_id, current_heading, text))

        passages: list[PolicyPassage] = []
        sequence = 1

        for sec_id, heading, text in sections:
            chunks = self._chunk_section_text(text)
            for chunk_text in chunks:
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                passage_id = f"{snapshot_id}:p{sequence:03d}"
                passages.append(
                    PolicyPassage(
                        id=passage_id,
                        snapshot_id=snapshot_id,
                        source_id=source_id,
                        section_identifier=sec_id,
                        heading=heading,
                        text=chunk_text,
                        sequence=sequence,
                        char_offset_start=0,
                        char_offset_end=len(chunk_text),
                        content_hash=chunk_hash,
                    )
                )
                sequence += 1

        return passages

    def _chunk_section_text(self, text: str) -> list[str]:
        """Split section text into chunks <= max_passage_chars at paragraph boundaries."""
        if len(text) <= self.max_passage_chars:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > self.max_passage_chars:
                # Paragraph itself is too large, split at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk) + len(sentence) + 1 <= self.max_passage_chars:
                        current_chunk = f"{current_chunk} {sentence}".strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence[: self.max_passage_chars]
            elif len(current_chunk) + len(para) + 2 <= self.max_passage_chars:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text[: self.max_passage_chars]]


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
        passages=passages,
        metadata=snapshot_metadata,
    )

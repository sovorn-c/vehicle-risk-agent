"""Domain models for Policy Source, Snapshot, Passage, and Citations."""

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AuthorityClassification(StrEnum):
    """Hierarchy classification of official policy authorities in New Zealand."""

    PRIMARY_LEGISLATION = "PRIMARY_LEGISLATION"
    REGULATION = "REGULATION"
    REGULATOR_GUIDANCE = "REGULATOR_GUIDANCE"
    CONSUMER_PROTECTION_GUIDANCE = "CONSUMER_PROTECTION_GUIDANCE"
    STATUTORY_CODE = "STATUTORY_CODE"
    OFFICIAL_GUIDANCE = "OFFICIAL_GUIDANCE"


class SourceStatus(StrEnum):
    """Lifecycle status of a Policy Source."""

    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


class ValidationOutcome(StrEnum):
    """Structural validation status of captured policy content."""

    VALID = "VALID"
    INVALID = "INVALID"


class FrozenDict(dict[str, Any]):
    """JSON-compatible recursively immutable dictionary."""

    def _raise_mutation(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("metadata is immutable")

    __setitem__ = _raise_mutation
    __delitem__ = _raise_mutation
    clear = _raise_mutation
    pop = _raise_mutation
    popitem = _raise_mutation  # type: ignore[assignment]
    setdefault = _raise_mutation
    update = _raise_mutation
    __ior__ = _raise_mutation  # type: ignore[assignment]


def _freeze_value(value: Any) -> Any:
    """Freeze standard mutable containers while preserving JSON serialization."""
    if isinstance(value, dict):
        return FrozenDict({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_metadata(value: dict[str, Any]) -> FrozenDict:
    """Freeze a metadata mapping recursively."""
    frozen = _freeze_value(value)
    assert isinstance(frozen, FrozenDict)
    return frozen


class PolicySource(BaseModel):
    """Official publication registered in the policy knowledge base."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=256)
    issuing_authority: str = Field(min_length=1, max_length=256)
    jurisdiction: str = Field(default="NZ", min_length=2, max_length=2)
    canonical_origin: str = Field(min_length=1, max_length=1024)
    authority_classification: AuthorityClassification
    reuse_terms: str = Field(min_length=1, max_length=512)
    expected_update_cadence: str = Field(min_length=1, max_length=64)
    status: SourceStatus = SourceStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PolicyPassage(BaseModel):
    """Immutable section-level text passage attributable to a snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=256)
    snapshot_id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=128)
    section_identifier: str = Field(min_length=1, max_length=128)
    heading: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=1200)
    sequence: int = Field(ge=1)
    char_offset_start: int = Field(ge=0, default=0)
    char_offset_end: int = Field(ge=0, default=0)
    content_hash: str = Field(min_length=64, max_length=64)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, v: str) -> str:
        """Ensure hash is valid hex."""
        if not all(c in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("content_hash must be a valid hex sha256 string")
        return v.lower()

    @model_validator(mode="after")
    def verify_text_hash(self) -> "PolicyPassage":
        """Verify that the declared content hash matches the passage text."""
        computed = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.content_hash != computed:
            raise ValueError(
                f"content_hash does not match text sha256 "
                f"(expected {computed}, got {self.content_hash})"
            )
        return self


class PolicySnapshot(BaseModel):
    """Immutable point-in-time capture of a Policy Source with attributable passages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=128)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    effective_date: datetime | None = None
    publication_date: datetime | None = None
    content_hash: str = Field(min_length=64, max_length=64)
    raw_content: str = Field(min_length=1)
    parser_version: str = Field(default="policy-parser-v1", min_length=1, max_length=64)
    validation_outcome: ValidationOutcome = ValidationOutcome.VALID
    passages: tuple[PolicyPassage, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _context: Any) -> None:
        """Freeze metadata after Pydantic accepts the JSON-compatible mapping."""
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @model_validator(mode="after")
    def verify_content_hash(self) -> "PolicySnapshot":
        """Verify that the declared content_hash matches sha256 of raw_content."""
        computed = hashlib.sha256(self.raw_content.encode("utf-8")).hexdigest()
        if self.content_hash.lower() != computed.lower():
            raise ValueError(
                f"content_hash does not match raw_content sha256 "
                f"(expected {computed}, got {self.content_hash})"
            )
        return self


class PolicyCitation(BaseModel):
    """Attribution citation linking a report statement to a specific Policy Passage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str = Field(min_length=1, max_length=256)
    snapshot_id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=128)
    section_identifier: str = Field(min_length=1, max_length=128)
    heading: str = Field(min_length=1, max_length=256)
    source_title: str = Field(min_length=1, max_length=256)
    canonical_origin: str = Field(min_length=1, max_length=1024)
    text: str | None = Field(default=None, max_length=1200)
    content_hash: str | None = Field(default=None, min_length=64, max_length=64)
    reuse_terms: str | None = Field(default=None, max_length=512)

"""API request and response schemas for policy sources and snapshots."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from vehicle_risk_agent.policy.models import AuthorityClassification


class PolicySourceCreateRequest(BaseModel):
    """Schema for registering a new policy source."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=256)
    issuing_authority: str = Field(min_length=1, max_length=256)
    jurisdiction: str = Field(default="NZ", min_length=2, max_length=2)
    canonical_origin: str = Field(min_length=1, max_length=1024)
    authority_classification: AuthorityClassification
    reuse_terms: str = Field(min_length=1, max_length=512)
    expected_update_cadence: str = Field(min_length=1, max_length=64)


class PolicySourceResponse(BaseModel):
    """Schema for returning registered policy source details."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    issuing_authority: str
    jurisdiction: str
    canonical_origin: str
    authority_classification: str
    reuse_terms: str
    expected_update_cadence: str
    status: str
    created_at: datetime
    updated_at: datetime


class PolicySnapshotIngestRequest(BaseModel):
    """Schema for requesting policy snapshot ingestion."""

    model_config = ConfigDict(extra="forbid")

    raw_content: str = Field(min_length=1)
    effective_date: datetime | None = None
    publication_date: datetime | None = None


class PolicyPassageResponse(BaseModel):
    """Schema for an attributable policy passage."""

    model_config = ConfigDict(extra="forbid")

    id: str
    snapshot_id: str
    source_id: str
    section_identifier: str
    heading: str
    text: str
    sequence: int
    content_hash: str


class PolicySnapshotResponse(BaseModel):
    """Schema for returning policy snapshot details and passages."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    retrieved_at: datetime
    effective_date: datetime | None = None
    publication_date: datetime | None = None
    content_hash: str
    parser_version: str
    validation_outcome: str
    passages: list[PolicyPassageResponse]

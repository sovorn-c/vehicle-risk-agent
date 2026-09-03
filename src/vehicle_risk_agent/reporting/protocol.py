"""Protocol and context definitions for report drafting adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.models import RiskResult

if TYPE_CHECKING:
    from vehicle_risk_agent.reporting.models import ReportDraft


class EvidenceItem(BaseModel):
    """Normalized evidence item representing an attributable observation or field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_name: str = Field(description="Canonical evidence field name")
    value: Any = Field(description="Observed canonical value")
    observation_id: str = Field(
        default="", description="Unique source observation identifier if available"
    )
    source_system: str = Field(default="", description="Originating register source")
    source_record_id: str = Field(default="", description="Record ID within source register")
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp observation was retrieved",
    )
    is_synthetic: bool = Field(
        default=False, description="Flag indicating synthetic demonstration provenance"
    )
    confidence_score: int | None = Field(
        default=None, ge=0, le=100, description="Field-level confidence score if evaluated"
    )
    explanation: str | None = Field(
        default=None, description="Human-readable field explanation or conflict rationale"
    )


class ReportDraftingContext(BaseModel):
    """Immutable input context required to draft an auditable risk report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: str = Field(description="Unique assessment identifier")
    run_number: int = Field(ge=1, description="Assessment run sequence number")
    vehicle_id: str = Field(default="", description="Canonical vehicle identifier")
    vin: str = Field(default="", description="Vehicle Identification Number")
    risk_result: RiskResult = Field(description="Calculated or withheld RiskResult")
    evidence_snapshot: VehicleEvidenceSnapshot | None = Field(
        default=None, description="Captured point-in-time evidence snapshot"
    )
    evidence_items: tuple[EvidenceItem, ...] = Field(
        default_factory=tuple, description="Normalized discrete evidence items"
    )
    policy_citations: tuple[PolicyCitation, ...] = Field(
        default_factory=tuple, description="Retrieved attributable policy citations"
    )
    drafter_id: str = Field(
        default="offline-v1", description="Identifier of drafting engine / adapter"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Supplementary contextual metadata"
    )

    @model_validator(mode="after")
    def sync_vin_and_vehicle_id(self) -> ReportDraftingContext:
        if not self.vin and self.vehicle_id:
            object.__setattr__(self, "vin", self.vehicle_id)
        elif not self.vehicle_id and self.vin:
            object.__setattr__(self, "vehicle_id", self.vin)
        elif (
            not self.vin
            and not self.vehicle_id
            and self.evidence_snapshot is not None
            and self.evidence_snapshot.vin
        ):
            object.__setattr__(self, "vin", self.evidence_snapshot.vin)
            object.__setattr__(self, "vehicle_id", self.evidence_snapshot.vin)
        return self

    def is_synthetic_context(self) -> bool:
        """Deterministically detect whether any input contains synthetic demonstration data."""
        if self.evidence_snapshot is not None:
            if self.evidence_snapshot.synthetic_notice:
                return True
            for prov_list in self.evidence_snapshot.field_provenance.values():
                if any(p.synthetic for p in prov_list):
                    return True
        if any(item.is_synthetic for item in self.evidence_items):
            return True
        return bool(self.metadata.get("is_synthetic", False))

    def get_canonical_field(self, field_name: str, default: Any = None) -> Any:
        """Safely retrieve a canonical field value from snapshot or evidence items."""
        if self.evidence_snapshot is not None:
            val = self.evidence_snapshot.canonical_fields.get(field_name)
            if val is not None:
                return val
        for item in self.evidence_items:
            if item.field_name == field_name and item.value is not None:
                return item.value
        return default

    def get_evidence_refs_for_field(self, field_name: str) -> tuple[str, ...]:
        """Extract sorted observation IDs supporting a given field."""
        if self.evidence_snapshot is not None and self.evidence_snapshot.field_provenance:
            provs = self.evidence_snapshot.field_provenance.get(field_name, ())
            refs = [p.observation_id for p in provs if p.observation_id]
            if refs:
                return tuple(sorted(set(refs)))
        refs = [
            item.observation_id
            for item in self.evidence_items
            if item.field_name == field_name and item.observation_id
        ]
        return tuple(sorted(set(refs)))


@runtime_checkable
class ReportDraftingProtocol(Protocol):
    """Asynchronous protocol implemented by report drafting adapters."""

    async def draft_report(self, context: ReportDraftingContext) -> ReportDraft:
        """Draft a structured, grounded report from the supplied context."""
        ...

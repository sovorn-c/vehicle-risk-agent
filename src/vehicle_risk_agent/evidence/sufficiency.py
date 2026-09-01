"""Deterministic evaluation of Required Evidence sufficiency for risk policy input."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from vehicle_risk_agent.evidence.models import ConflictState, SafeError
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot

REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = ("ppsr_result", "stolen_status", "writeoff_status")


class SufficiencyOutcome(StrEnum):
    """Categorical outcome of vehicle evidence sufficiency evaluation."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNAVAILABLE = "UNAVAILABLE"


class MissingEvidenceReason(StrEnum):
    """Specific cause for an incomplete required evidence field."""

    UNKNOWN = "UNKNOWN"
    UNRESOLVED = "UNRESOLVED"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"
    ABSENT = "ABSENT"
    LOOKUP_FAILED = "LOOKUP_FAILED"


class MissingEvidenceFinding(BaseModel):
    """Attributable finding recording an incomplete or unverified required evidence field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_name: str = Field(description="Target required evidence field name")
    reason: MissingEvidenceReason = Field(description="Specific cause for missing evidence")
    details: str = Field(description="Human-readable explanation of the missing evidence condition")


class EvidenceSufficiencyResult(BaseModel):
    """Immutable result of evaluating evidence completeness against Risk Policy requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: SufficiencyOutcome = Field(description="COMPLETE, INCOMPLETE, or UNAVAILABLE outcome")
    is_sufficient: bool = Field(description="True only when outcome is COMPLETE")
    missing_findings: tuple[MissingEvidenceFinding, ...] = Field(
        default_factory=tuple, description="Deterministic ordered list of missing evidence findings"
    )


class IncompleteAssessmentReport(BaseModel):
    """Prohibits score or band fields when evidence sufficiency outcome is INCOMPLETE."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    assessment_id: str = Field(description="Unique assessment identifier")
    run_number: int = Field(ge=1, description="Assessment run sequence number")
    vin: str = Field(description="Canonical 17-character VIN")
    outcome: SufficiencyOutcome = Field(default=SufficiencyOutcome.INCOMPLETE)
    missing_findings: tuple[MissingEvidenceFinding, ...] = Field(
        description="Attributable ordered missing evidence findings"
    )


def evaluate_evidence_sufficiency(
    snapshot: VehicleEvidenceSnapshot | None,
    failure_error: SafeError | None = None,
) -> EvidenceSufficiencyResult:
    """Deterministically evaluate if snapshot satisfies all Required Evidence fields."""
    if snapshot is None:
        details = (
            failure_error.message if failure_error else "Vehicle lookup failed or was unavailable."
        )
        finding = MissingEvidenceFinding(
            field_name="vehicle_lookup",
            reason=MissingEvidenceReason.LOOKUP_FAILED,
            details=details,
        )
        return EvidenceSufficiencyResult(
            outcome=SufficiencyOutcome.UNAVAILABLE,
            is_sufficient=False,
            missing_findings=(finding,),
        )

    findings: list[MissingEvidenceFinding] = []

    # Map unresolved conflicts by field
    unresolved_conflict_fields = {
        c.field_name: c for c in snapshot.conflicts if c.state != ConflictState.RESOLVED
    }

    for field in REQUIRED_EVIDENCE_FIELDS:
        if field in unresolved_conflict_fields:
            c = unresolved_conflict_fields[field]
            findings.append(
                MissingEvidenceFinding(
                    field_name=field,
                    reason=MissingEvidenceReason.UNRESOLVED_CONFLICT,
                    details=(
                        f"Field '{field}' has unresolved conflict between "
                        f"{len(c.conflicting_candidates)} competing candidate sources."
                    ),
                )
            )
        elif field not in snapshot.canonical_fields:
            findings.append(
                MissingEvidenceFinding(
                    field_name=field,
                    reason=MissingEvidenceReason.ABSENT,
                    details=f"Required field '{field}' is absent from upstream canonical fields.",
                )
            )
        else:
            val = snapshot.canonical_fields[field]
            normalized = val.strip().upper() if isinstance(val, str) else None
            if val is None or normalized in (None, "", "UNKNOWN"):
                findings.append(
                    MissingEvidenceFinding(
                        field_name=field,
                        reason=MissingEvidenceReason.UNKNOWN,
                        details=(
                            f"Required field '{field}' is absent, blank, or unknown "
                            "in upstream evidence."
                        ),
                    )
                )
            elif normalized == "UNRESOLVED":
                findings.append(
                    MissingEvidenceFinding(
                        field_name=field,
                        reason=MissingEvidenceReason.UNRESOLVED,
                        details=(f"Required field '{field}' is unresolved in upstream evidence."),
                    )
                )

    # Sort deterministically by field name
    sorted_findings = tuple(sorted(findings, key=lambda f: f.field_name))

    if sorted_findings:
        return EvidenceSufficiencyResult(
            outcome=SufficiencyOutcome.INCOMPLETE,
            is_sufficient=False,
            missing_findings=sorted_findings,
        )

    return EvidenceSufficiencyResult(
        outcome=SufficiencyOutcome.COMPLETE,
        is_sufficient=True,
        missing_findings=(),
    )

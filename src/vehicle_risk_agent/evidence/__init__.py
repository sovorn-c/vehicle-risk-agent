"""Vehicle evidence contracts, MCP adapter, sufficiency evaluation, and persistence."""

from vehicle_risk_agent.evidence.history import collect_vehicle_history
from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    FieldExplanationResult,
    FieldOutcome,
    ProvenanceLink,
    SafeError,
    SafeErrorCategory,
    SourceObservationResponse,
    VehicleRevisionResponse,
)
from vehicle_risk_agent.evidence.parallel import (
    DuplicateEvidenceKeyError,
    explain_fields_in_parallel,
    merge_field_explanations,
)
from vehicle_risk_agent.evidence.snapshot import (
    VehicleEvidenceRepository,
    VehicleEvidenceSnapshot,
    create_evidence_snapshot,
)
from vehicle_risk_agent.evidence.sufficiency import (
    REQUIRED_EVIDENCE_FIELDS,
    EvidenceSufficiencyResult,
    IncompleteAssessmentReport,
    MissingEvidenceFinding,
    MissingEvidenceReason,
    SufficiencyOutcome,
    evaluate_evidence_sufficiency,
)

__all__ = [
    "CandidateValue",
    "ConfidenceAssessment",
    "ConfidenceBand",
    "ConflictState",
    "DuplicateEvidenceKeyError",
    "EvidenceSufficiencyResult",
    "FieldConflict",
    "FieldExplanationResult",
    "FieldOutcome",
    "IncompleteAssessmentReport",
    "MissingEvidenceFinding",
    "MissingEvidenceReason",
    "ProvenanceLink",
    "REQUIRED_EVIDENCE_FIELDS",
    "SafeError",
    "SafeErrorCategory",
    "SourceObservationResponse",
    "SufficiencyOutcome",
    "VehicleEvidenceRepository",
    "VehicleEvidenceSnapshot",
    "VehicleRevisionResponse",
    "collect_vehicle_history",
    "create_evidence_snapshot",
    "evaluate_evidence_sufficiency",
    "explain_fields_in_parallel",
    "merge_field_explanations",
]

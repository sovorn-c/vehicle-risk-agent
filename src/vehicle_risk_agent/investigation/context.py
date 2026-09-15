"""Build minimized, non-authoritative context for proposal generation."""

from __future__ import annotations

from collections.abc import Iterable

from vehicle_risk_agent.evidence.models import FieldExplanationResult, VehicleRevisionResponse
from vehicle_risk_agent.investigation.models import (
    ALLOWED_EVIDENCE_TARGETS,
    InvestigationContext,
)
from vehicle_risk_agent.policy.models import PolicyCitation


def build_investigation_context(
    *,
    revision: VehicleRevisionResponse,
    questions: Iterable[str] = (),
    evidence_targets: Iterable[str] = (),
    policy_citations: Iterable[PolicyCitation] = (),
    prior_results: Iterable[FieldExplanationResult] = (),
) -> InvestigationContext:
    """Project authoritative inputs into bounded summaries without raw payloads."""
    targets = tuple(dict.fromkeys(item.strip().lower() for item in evidence_targets))
    if not targets:
        targets = tuple(sorted(set(revision.canonical_fields) & ALLOWED_EVIDENCE_TARGETS))[:5]
    if any(target not in ALLOWED_EVIDENCE_TARGETS for target in targets):
        raise ValueError("evidence target is not allowed")

    summaries: list[str] = [f"snapshot_conflicts={len(revision.conflicts)}"]
    for target in targets:
        present = target in revision.canonical_fields
        provenance = revision.field_provenance.get(target, ())
        references = ",".join(link.observation_id for link in provenance[:5]) or "none"
        status = "present" if present else "absent"
        summaries.append(
            f"field={target}; status={status}; revision={revision.revision_number}; "
            f"confidence={revision.confidence.score}; references={references}"
        )

    result_summaries = [
        (
            f"field={result.field_name}; outcome={result.outcome.value}; "
            f"revision={result.revision_number}; confidence={result.field_confidence_score}; "
            f"references="
            f"{','.join(link.observation_id for link in result.provenance[:5]) or 'none'}"
        )
        for result in prior_results
    ]
    stable_references = [revision.revision_id, revision.material_hash]
    stable_references.extend(
        link.observation_id for links in revision.field_provenance.values() for link in links[:5]
    )
    stable_references.extend(citation.passage_id for citation in policy_citations)
    return InvestigationContext(
        questions=tuple(questions)[:5],
        evidence_targets=targets[:5],
        evidence_summaries=tuple(summaries[:20]),
        prior_result_summaries=tuple(result_summaries[:3]),
        stable_references=tuple(dict.fromkeys(stable_references))[:50],
    )

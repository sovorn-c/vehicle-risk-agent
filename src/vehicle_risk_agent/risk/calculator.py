"""Pure deterministic risk scoring calculator and invariant proofs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot
from vehicle_risk_agent.evidence.sufficiency import (
    EvidenceSufficiencyResult,
    SufficiencyOutcome,
    evaluate_evidence_sufficiency,
)
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    MandatoryFinding,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskPolicy,
    RiskResult,
)

POSITIVE_PPSR_VALUES = frozenset(
    {
        "MATCH",
        "FINANCE_REGISTERED",
        "SECURITY_INTEREST_FOUND",
        "YES",
        "ACTIVE",
        "REGISTERED_SECURITY_INTEREST",
    }
)
POSITIVE_STOLEN_VALUES = frozenset({"LISTED", "STOLEN", "REPORTED_STOLEN", "YES", "ACTIVE"})
REPAIRABLE_WRITEOFF_VALUES = frozenset({"REPAIRABLE", "REPAIRABLE_WRITEOFF"})
STATUTORY_WRITEOFF_VALUES = frozenset(
    {"STATUTORY", "STATUTORY_WRITEOFF", "DEREGISTERED", "NON_REPAIRABLE"}
)

FACTOR_CITATION_KEYWORDS: dict[RiskFactor, tuple[str, ...]] = {
    RiskFactor.MATCH: (
        "ppsr",
        "personal property",
        "security interest",
        "financing statement",
        "encumbrance",
    ),
    RiskFactor.LISTED: ("stolen", "police", "theft", "recovery", "title"),
    RiskFactor.REPAIRABLE: ("repairable", "damaged", "write-off", "inspection", "certification"),
    RiskFactor.STATUTORY: (
        "statutory",
        "non-repairable",
        "dismantling",
        "deregistration",
        "written-off",
    ),
}


def _extract_evidence_refs(
    snapshot: VehicleEvidenceSnapshot | None, field_name: str
) -> tuple[str, ...]:
    """Extract ordered observation IDs from snapshot provenance for a field."""
    if snapshot is None or not snapshot.field_provenance:
        return ()
    provenance_links = snapshot.field_provenance.get(field_name, ())
    refs = [p.observation_id for p in provenance_links if p.observation_id]
    return tuple(sorted(set(refs)))


def _match_policy_citations(
    factor: RiskFactor, citations: Sequence[PolicyCitation]
) -> tuple[str, ...]:
    """Find matching Policy Passage IDs for a factor based on keywords and headings."""
    matched_ids: set[str] = set()
    keywords = FACTOR_CITATION_KEYWORDS.get(factor, ())

    for citation in citations:
        haystack = (
            f"{citation.source_title} {citation.heading} {citation.section_identifier}".lower()
        )
        if any(kw in haystack for kw in keywords):
            matched_ids.add(citation.passage_id)

    return tuple(sorted(matched_ids))


def compute_calculation_hash(
    policy_id: str,
    policy_version: str,
    policy_hash: str,
    canonical_fields: dict[str, Any],
    outcome: AssessmentOutcome,
    score: int | None,
    band: RiskBand | None,
    raw_score: int | None,
    triggered_factors: Sequence[RiskFactor],
) -> str:
    """Compute deterministic SHA-256 fingerprint for a calculation's inputs and outputs."""
    payload = {
        "policy_id": policy_id,
        "policy_version": policy_version,
        "policy_hash": policy_hash,
        "canonical_fields": {k: str(v) for k, v in sorted(canonical_fields.items())},
        "outcome": outcome.value,
        "score": score,
        "band": band.value if band else None,
        "raw_score": raw_score,
        "triggered_factors": [
            str(f.value if isinstance(f, RiskFactor) else f) for f in sorted(triggered_factors)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calculate_risk_result(
    policy: RiskPolicy,
    snapshot: VehicleEvidenceSnapshot | None,
    sufficiency: EvidenceSufficiencyResult | None = None,
    assessment_id: str | None = None,
    run_number: int | None = None,
    policy_citations: Sequence[PolicyCitation] = (),
    calculated_at: datetime | None = None,
) -> RiskResult:
    """Deterministically compute risk score, band, factors, and mandatory findings."""
    aid = (
        snapshot.assessment_id if snapshot is not None else (assessment_id or "unknown-assessment")
    )
    rnum = snapshot.run_number if snapshot is not None else (run_number or 1)
    calc_time = calculated_at or (
        snapshot.collected_at if snapshot is not None else datetime.now(UTC)
    )

    if sufficiency is None:
        sufficiency = evaluate_evidence_sufficiency(snapshot)

    # If evidence is not complete, strictly withhold scoring
    if snapshot is None or sufficiency.outcome != SufficiencyOutcome.COMPLETE:
        missing = sufficiency.missing_findings if sufficiency is not None else ()
        calc_hash = compute_calculation_hash(
            policy_id=policy.id,
            policy_version=policy.version,
            policy_hash=policy.policy_hash,
            canonical_fields=snapshot.canonical_fields if snapshot is not None else {},
            outcome=AssessmentOutcome.INCOMPLETE,
            score=None,
            band=None,
            raw_score=None,
            triggered_factors=(),
        )
        return RiskResult(
            assessment_id=aid,
            run_number=rnum,
            policy_id=policy.id,
            policy_version=policy.version,
            score=None,
            band=None,
            raw_score=None,
            is_incomplete=True,
            outcome=AssessmentOutcome.INCOMPLETE,
            factors=(),
            findings=(),
            missing_evidence=missing,
            missing_findings=missing,
            calculated_at=calc_time,
            calculation_hash=calc_hash,
        )

    fields = snapshot.canonical_fields
    factor_results: list[RiskFactorResult] = []
    findings: list[MandatoryFinding] = []
    triggered_factors: list[RiskFactor] = []

    # 1. MATCH factor (PPSR)
    ppsr_val = str(fields.get("ppsr_result", "")).strip().upper()
    match_triggered = ppsr_val in POSITIVE_PPSR_VALUES
    match_weight = policy.factor_weights.get(RiskFactor.MATCH, 30)
    match_evidence_refs = _extract_evidence_refs(snapshot, "ppsr_result")
    match_citation_refs = _match_policy_citations(RiskFactor.MATCH, policy_citations)

    factor_results.append(
        RiskFactorResult(
            factor=RiskFactor.MATCH,
            weight=match_weight,
            triggered=match_triggered,
            evidence_field="ppsr_result",
            evidence_value=fields.get("ppsr_result"),
            rationale="Registered security interest identified on PPSR register"
            if match_triggered
            else "No registered security interest found",
            score_contribution=match_weight if match_triggered else 0,
            evidence_refs=match_evidence_refs,
            policy_citation_refs=match_citation_refs,
        )
    )
    if match_triggered:
        triggered_factors.append(RiskFactor.MATCH)
        findings.append(
            MandatoryFinding(
                finding_id=f"finding-match-{aid}-{rnum}",
                factor=RiskFactor.MATCH,
                title="Registered Security Interest Detected",
                description=(
                    "PPSR record indicates an active security interest or "
                    "financial encumbrance registered against the vehicle."
                ),
                evidence_field="ppsr_result",
                evidence_value=fields.get("ppsr_result"),
                weight=match_weight,
                severity=RiskBand.MEDIUM,
                evidence_refs=match_evidence_refs,
                policy_citation_refs=match_citation_refs,
            )
        )

    # 2. LISTED factor (Stolen)
    stolen_val = str(fields.get("stolen_status", "")).strip().upper()
    listed_triggered = stolen_val in POSITIVE_STOLEN_VALUES
    listed_weight = policy.factor_weights.get(RiskFactor.LISTED, 45)
    listed_evidence_refs = _extract_evidence_refs(snapshot, "stolen_status")
    listed_citation_refs = _match_policy_citations(RiskFactor.LISTED, policy_citations)

    factor_results.append(
        RiskFactorResult(
            factor=RiskFactor.LISTED,
            weight=listed_weight,
            triggered=listed_triggered,
            evidence_field="stolen_status",
            evidence_value=fields.get("stolen_status"),
            rationale="Vehicle is actively listed as stolen on police register"
            if listed_triggered
            else "Vehicle is not listed as stolen",
            score_contribution=listed_weight if listed_triggered else 0,
            evidence_refs=listed_evidence_refs,
            policy_citation_refs=listed_citation_refs,
        )
    )
    if listed_triggered:
        triggered_factors.append(RiskFactor.LISTED)
        findings.append(
            MandatoryFinding(
                finding_id=f"finding-listed-{aid}-{rnum}",
                factor=RiskFactor.LISTED,
                title="Active Stolen Vehicle Record",
                description="Police register indicates the vehicle is currently listed as stolen.",
                evidence_field="stolen_status",
                evidence_value=fields.get("stolen_status"),
                weight=listed_weight,
                severity=RiskBand.HIGH,
                evidence_refs=listed_evidence_refs,
                policy_citation_refs=listed_citation_refs,
            )
        )

    # 3. Write-off factors (REPAIRABLE vs STATUTORY)
    writeoff_val = str(fields.get("writeoff_status", "")).strip().upper()
    statutory_triggered = writeoff_val in STATUTORY_WRITEOFF_VALUES
    repairable_triggered = (not statutory_triggered) and (
        writeoff_val in REPAIRABLE_WRITEOFF_VALUES
    )

    # REPAIRABLE
    repairable_weight = policy.factor_weights.get(RiskFactor.REPAIRABLE, 20)
    repairable_evidence_refs = _extract_evidence_refs(snapshot, "writeoff_status")
    repairable_citation_refs = _match_policy_citations(RiskFactor.REPAIRABLE, policy_citations)

    factor_results.append(
        RiskFactorResult(
            factor=RiskFactor.REPAIRABLE,
            weight=repairable_weight,
            triggered=repairable_triggered,
            evidence_field="writeoff_status",
            evidence_value=fields.get("writeoff_status"),
            rationale="Vehicle recorded with repairable write-off status"
            if repairable_triggered
            else "No repairable write-off record",
            score_contribution=repairable_weight if repairable_triggered else 0,
            evidence_refs=repairable_evidence_refs,
            policy_citation_refs=repairable_citation_refs,
        )
    )
    if repairable_triggered:
        triggered_factors.append(RiskFactor.REPAIRABLE)
        findings.append(
            MandatoryFinding(
                finding_id=f"finding-repairable-{aid}-{rnum}",
                factor=RiskFactor.REPAIRABLE,
                title="Repairable Write-Off History",
                description="Vehicle has an official repairable write-off record.",
                evidence_field="writeoff_status",
                evidence_value=fields.get("writeoff_status"),
                weight=repairable_weight,
                severity=RiskBand.MEDIUM,
                evidence_refs=repairable_evidence_refs,
                policy_citation_refs=repairable_citation_refs,
            )
        )

    # STATUTORY
    statutory_weight = policy.factor_weights.get(RiskFactor.STATUTORY, 40)
    statutory_evidence_refs = _extract_evidence_refs(snapshot, "writeoff_status")
    statutory_citation_refs = _match_policy_citations(RiskFactor.STATUTORY, policy_citations)

    factor_results.append(
        RiskFactorResult(
            factor=RiskFactor.STATUTORY,
            weight=statutory_weight,
            triggered=statutory_triggered,
            evidence_field="writeoff_status",
            evidence_value=fields.get("writeoff_status"),
            rationale="Vehicle recorded with statutory/non-repairable write-off or deregistration"
            if statutory_triggered
            else "No statutory write-off record",
            score_contribution=statutory_weight if statutory_triggered else 0,
            evidence_refs=statutory_evidence_refs,
            policy_citation_refs=statutory_citation_refs,
        )
    )
    if statutory_triggered:
        triggered_factors.append(RiskFactor.STATUTORY)
        findings.append(
            MandatoryFinding(
                finding_id=f"finding-statutory-{aid}-{rnum}",
                factor=RiskFactor.STATUTORY,
                title="Statutory Write-Off / Deregistered Record",
                description=(
                    "Vehicle is recorded with statutory/non-repairable write-off or deregistration."
                ),
                evidence_field="writeoff_status",
                evidence_value=fields.get("writeoff_status"),
                weight=statutory_weight,
                severity=RiskBand.HIGH,
                evidence_refs=statutory_evidence_refs,
                policy_citation_refs=statutory_citation_refs,
            )
        )

    # Score calculation & Cap
    raw_score = sum(fr.weight for fr in factor_results if fr.triggered)
    capped_score = min(raw_score, policy.score_cap)

    # Risk Band Determination
    matched_band = RiskBand.LOW
    for b_def in sorted(policy.risk_bands, key=lambda b: b.min_score):
        if b_def.min_score <= capped_score <= b_def.max_score:
            matched_band = b_def.band
            break

    # Calculation Hash
    calc_hash = compute_calculation_hash(
        policy_id=policy.id,
        policy_version=policy.version,
        policy_hash=policy.policy_hash,
        canonical_fields=fields,
        outcome=AssessmentOutcome.SCORED,
        score=capped_score,
        band=matched_band,
        raw_score=raw_score,
        triggered_factors=triggered_factors,
    )

    return RiskResult(
        id=str(uuid4()),
        assessment_id=aid,
        run_number=rnum,
        policy_id=policy.id,
        policy_version=policy.version,
        score=capped_score,
        band=matched_band,
        raw_score=raw_score,
        is_incomplete=False,
        outcome=AssessmentOutcome.SCORED,
        factors=tuple(sorted(factor_results, key=lambda fr: fr.factor.value)),
        findings=tuple(sorted(findings, key=lambda f: f.factor.value)),
        missing_evidence=(),
        missing_findings=(),
        calculated_at=calc_time,
        calculation_hash=calc_hash,
    )


def calculate_risk_from_snapshot(
    policy: RiskPolicy,
    snapshot: VehicleEvidenceSnapshot,
    policy_citations: Sequence[PolicyCitation] = (),
    sufficiency: EvidenceSufficiencyResult | None = None,
) -> RiskResult:
    """Convenience wrapper calculating risk result directly from VehicleEvidenceSnapshot."""
    return calculate_risk_result(
        policy=policy,
        snapshot=snapshot,
        sufficiency=sufficiency,
        policy_citations=policy_citations,
    )

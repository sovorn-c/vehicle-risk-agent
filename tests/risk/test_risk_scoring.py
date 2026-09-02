"""Unit tests for pure deterministic risk calculation engine, invariants, and proofs."""

import itertools
import random
from datetime import UTC, datetime

from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    ProvenanceLink,
)
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot
from vehicle_risk_agent.evidence.sufficiency import (
    MissingEvidenceReason,
)
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.calculator import (
    calculate_risk_from_snapshot,
    calculate_risk_result,
)
from vehicle_risk_agent.risk.models import (
    AssessmentOutcome,
    RiskBand,
    RiskFactor,
    build_risk_policy_v1,
)


def _build_snapshot(
    ppsr_result: str = "NO_MATCH",
    stolen_status: str = "NOT_STOLEN",
    writeoff_status: str = "NONE",
    assessment_id: str = "asmt-001",
    run_number: int = 1,
    conflicts: tuple[FieldConflict, ...] = (),
) -> VehicleEvidenceSnapshot:
    """Helper to build a deterministic VehicleEvidenceSnapshot."""
    fields = {
        "ppsr_result": ppsr_result,
        "stolen_status": stolen_status,
        "writeoff_status": writeoff_status,
        "make": "Toyota",
        "model": "Corolla",
        "year": 2018,
    }
    provenance: dict[str, tuple[ProvenanceLink, ...]] = {
        "ppsr_result": (
            ProvenanceLink(
                observation_id="obs-ppsr-001",
                source_system="PPSR",
                source_record_id="ppsr-rec-001",
                retrieved_at=datetime.now(UTC),
            ),
        ),
        "stolen_status": (
            ProvenanceLink(
                observation_id="obs-stolen-001",
                source_system="POLICE",
                source_record_id="stolen-rec-001",
                retrieved_at=datetime.now(UTC),
            ),
        ),
        "writeoff_status": (
            ProvenanceLink(
                observation_id="obs-writeoff-001",
                source_system="NZTA",
                source_record_id="writeoff-rec-001",
                retrieved_at=datetime.now(UTC),
            ),
        ),
    }
    return VehicleEvidenceSnapshot(
        assessment_id=assessment_id,
        run_number=run_number,
        vin="7AT0BJ12345678901",
        revision_id="rev-001",
        revision_number=1,
        material_hash="a" * 64,
        canonical_fields=fields,
        field_provenance=provenance,
        conflicts=conflicts,
        confidence=ConfidenceAssessment(
            score=90, band=ConfidenceBand.HIGH, rule_version="v1", explanation="High quality data"
        ),
        as_of=datetime.now(UTC),
        published_at=datetime.now(UTC),
    )


def _sample_citations() -> tuple[PolicyCitation, ...]:
    """Helper to provide realistic Policy Citations for testing attribution."""
    return (
        PolicyCitation(
            passage_id="pass-ppsr-01",
            snapshot_id="snap-ppsr-01",
            source_id="src-ppsa-1999",
            section_identifier="s52",
            heading="Personal Property Securities Act 1999 - Security Interests",
            source_title="Personal Property Securities Act 1999",
            canonical_origin="https://www.legislation.govt.nz/act/public/1999/0126/latest/DLM45900.html",
        ),
        PolicyCitation(
            passage_id="pass-stolen-01",
            snapshot_id="snap-police-01",
            source_id="src-police-stolen",
            section_identifier="s2",
            heading="Police Stolen Vehicle Register - Recovery and Title",
            source_title="New Zealand Police Stolen Vehicle Guidance",
            canonical_origin="https://www.police.govt.nz/stolen-vehicles",
        ),
        PolicyCitation(
            passage_id="pass-repairable-01",
            snapshot_id="snap-nzta-01",
            source_id="src-nzta-writeoff",
            section_identifier="s14",
            heading="NZTA Damaged and Written-off Vehicles - Repairable Certification",
            source_title="NZTA Vehicle Inspection and Certification Rules",
            canonical_origin="https://www.nzta.govt.nz/vehicles/damaged-vehicles",
        ),
        PolicyCitation(
            passage_id="pass-statutory-01",
            snapshot_id="snap-nzta-02",
            source_id="src-nzta-writeoff",
            section_identifier="s15",
            heading="NZTA Statutory Non-Repairable Write-offs and Dismantling",
            source_title="NZTA Vehicle Inspection and Certification Rules",
            canonical_origin="https://www.nzta.govt.nz/vehicles/statutory-writeoffs",
        ),
    )


def test_clean_seed_scores_zero_low() -> None:
    """A clean vehicle with no adverse register entries scores 0 / LOW."""
    policy = build_risk_policy_v1()
    snapshot = _build_snapshot(
        ppsr_result="NO_MATCH",
        stolen_status="NOT_LISTED",
        writeoff_status="NONE",
    )

    result = calculate_risk_from_snapshot(policy=policy, snapshot=snapshot)

    assert result.is_incomplete is False
    assert result.outcome == AssessmentOutcome.SCORED
    assert result.score == 0
    assert result.raw_score == 0
    assert result.band == RiskBand.LOW
    assert len(result.findings) == 0
    assert len(result.missing_findings) == 0

    # All factors evaluated to not triggered
    for factor in result.factors:
        assert factor.triggered is False
        assert factor.score_contribution == 0


def test_exact_single_factor_scores_and_bands() -> None:
    """Test each single factor independently yields the exact point weight and correct band."""
    policy = build_risk_policy_v1()

    # 1. REPAIRABLE only (20 points -> MEDIUM 20-39)
    snap_repairable = _build_snapshot(writeoff_status="REPAIRABLE")
    res_repairable = calculate_risk_from_snapshot(policy, snap_repairable)
    assert res_repairable.score == 20
    assert res_repairable.raw_score == 20
    assert res_repairable.band == RiskBand.MEDIUM
    assert len(res_repairable.findings) == 1
    assert res_repairable.findings[0].factor == RiskFactor.REPAIRABLE
    assert res_repairable.findings[0].weight == 20

    # 2. MATCH only (30 points -> MEDIUM 20-39)
    snap_match = _build_snapshot(ppsr_result="MATCH")
    res_match = calculate_risk_from_snapshot(policy, snap_match)
    assert res_match.score == 30
    assert res_match.raw_score == 30
    assert res_match.band == RiskBand.MEDIUM
    assert len(res_match.findings) == 1
    assert res_match.findings[0].factor == RiskFactor.MATCH
    assert res_match.findings[0].weight == 30

    # 3. STATUTORY only (40 points -> HIGH 40-69)
    snap_statutory = _build_snapshot(writeoff_status="STATUTORY")
    res_statutory = calculate_risk_from_snapshot(policy, snap_statutory)
    assert res_statutory.score == 40
    assert res_statutory.raw_score == 40
    assert res_statutory.band == RiskBand.HIGH
    assert len(res_statutory.findings) == 1
    assert res_statutory.findings[0].factor == RiskFactor.STATUTORY
    assert res_statutory.findings[0].weight == 40

    # 4. LISTED only (45 points -> HIGH 40-69)
    snap_listed = _build_snapshot(stolen_status="LISTED")
    res_listed = calculate_risk_from_snapshot(policy, snap_listed)
    assert res_listed.score == 45
    assert res_listed.raw_score == 45
    assert res_listed.band == RiskBand.HIGH
    assert len(res_listed.findings) == 1
    assert res_listed.findings[0].factor == RiskFactor.LISTED
    assert res_listed.findings[0].weight == 45


def test_combined_factors_score_and_bands() -> None:
    """Test additive combinations of factors and resulting risk bands."""
    policy = build_risk_policy_v1()

    # MATCH (30) + REPAIRABLE (20) = 50 -> HIGH (40-69)
    res_50 = calculate_risk_from_snapshot(
        policy, _build_snapshot(ppsr_result="MATCH", writeoff_status="REPAIRABLE")
    )
    assert res_50.score == 50
    assert res_50.raw_score == 50
    assert res_50.band == RiskBand.HIGH
    assert len(res_50.findings) == 2

    # MATCH (30) + STATUTORY (40) = 70 -> CRITICAL (70-100)
    res_70 = calculate_risk_from_snapshot(
        policy, _build_snapshot(ppsr_result="MATCH", writeoff_status="STATUTORY")
    )
    assert res_70.score == 70
    assert res_70.raw_score == 70
    assert res_70.band == RiskBand.CRITICAL
    assert len(res_70.findings) == 2

    # LISTED (45) + MATCH (30) = 75 -> CRITICAL (70-100)
    res_75 = calculate_risk_from_snapshot(
        policy, _build_snapshot(stolen_status="LISTED", ppsr_result="MATCH")
    )
    assert res_75.score == 75
    assert res_75.raw_score == 75
    assert res_75.band == RiskBand.CRITICAL
    assert len(res_75.findings) == 2

    # LISTED (45) + STATUTORY (40) = 85 -> CRITICAL (70-100)
    res_85 = calculate_risk_from_snapshot(
        policy, _build_snapshot(stolen_status="LISTED", writeoff_status="STATUTORY")
    )
    assert res_85.score == 85
    assert res_85.raw_score == 85
    assert res_85.band == RiskBand.CRITICAL
    assert len(res_85.findings) == 2


def test_score_cap_at_100() -> None:
    """Test that sums exceeding 100 points are cleanly capped at 100 while preserving raw_score."""
    policy = build_risk_policy_v1()

    # MATCH (30) + LISTED (45) + STATUTORY (40) = 115 -> capped at 100
    res_115 = calculate_risk_from_snapshot(
        policy,
        _build_snapshot(
            ppsr_result="MATCH",
            stolen_status="LISTED",
            writeoff_status="STATUTORY",
        ),
    )
    assert res_115.raw_score == 115
    assert res_115.score == 100
    assert res_115.band == RiskBand.CRITICAL
    assert len(res_115.findings) == 3


def test_scoring_monotonicity_across_power_set() -> None:
    """Prove mathematical monotonicity: for any factor subsets A <= B, Score(A) <= Score(B)."""
    policy = build_risk_policy_v1()
    factors = [RiskFactor.MATCH, RiskFactor.LISTED, RiskFactor.REPAIRABLE, RiskFactor.STATUTORY]

    def score_for_subset(active_factors: set[RiskFactor]) -> int:
        ppsr = "MATCH" if RiskFactor.MATCH in active_factors else "NO_MATCH"
        stolen = "LISTED" if RiskFactor.LISTED in active_factors else "NOT_STOLEN"
        if RiskFactor.STATUTORY in active_factors:
            writeoff = "STATUTORY"
        elif RiskFactor.REPAIRABLE in active_factors:
            writeoff = "REPAIRABLE"
        else:
            writeoff = "NONE"

        snap = _build_snapshot(ppsr_result=ppsr, stolen_status=stolen, writeoff_status=writeoff)
        res = calculate_risk_from_snapshot(policy, snap)
        assert res.score is not None
        return res.score

    # Test all 16 subset pairs
    all_subsets = []
    for r in range(len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            all_subsets.append(set(combo))

    for sub_a in all_subsets:
        for sub_b in all_subsets:
            # If writeoff has both STATUTORY and REPAIRABLE, skip or test dominance
            if sub_a.issubset(sub_b):
                score_a = score_for_subset(sub_a)
                score_b = score_for_subset(sub_b)
                assert score_a <= score_b, (
                    f"Monotonicity violated: {sub_a} ({score_a}) vs {sub_b} ({score_b})"
                )


def test_scoring_order_independence_and_permutation_invariance() -> None:
    """Prove that shuffling field order or citations produces identical bit-for-bit result."""
    policy = build_risk_policy_v1()
    citations = _sample_citations()

    base_snapshot = _build_snapshot(
        ppsr_result="MATCH",
        stolen_status="LISTED",
        writeoff_status="STATUTORY",
    )

    base_result = calculate_risk_from_snapshot(
        policy=policy, snapshot=base_snapshot, policy_citations=citations
    )

    # Permute dictionary and citations 20 times
    for _ in range(20):
        shuffled_items = list(base_snapshot.canonical_fields.items())
        random.shuffle(shuffled_items)
        shuffled_fields = dict(shuffled_items)

        shuffled_citations = list(citations)
        random.shuffle(shuffled_citations)

        shuffled_snap = VehicleEvidenceSnapshot(
            assessment_id=base_snapshot.assessment_id,
            run_number=base_snapshot.run_number,
            vin=base_snapshot.vin,
            revision_id=base_snapshot.revision_id,
            revision_number=base_snapshot.revision_number,
            material_hash=base_snapshot.material_hash,
            canonical_fields=shuffled_fields,
            field_provenance=dict(base_snapshot.field_provenance),
            conflicts=base_snapshot.conflicts,
            confidence=base_snapshot.confidence,
            as_of=base_snapshot.as_of,
            published_at=base_snapshot.published_at,
        )

        perm_result = calculate_risk_from_snapshot(
            policy=policy, snapshot=shuffled_snap, policy_citations=shuffled_citations
        )

        assert perm_result.score == base_result.score
        assert perm_result.band == base_result.band
        assert perm_result.raw_score == base_result.raw_score
        assert perm_result.calculation_hash == base_result.calculation_hash
        assert [f.finding_id for f in perm_result.findings] == [
            f.finding_id for f in base_result.findings
        ]


def test_mandatory_findings_attributable_citations() -> None:
    """Verify generated findings contain observation IDs and policy citation references."""
    policy = build_risk_policy_v1()
    citations = _sample_citations()
    snapshot = _build_snapshot(
        ppsr_result="MATCH",
        stolen_status="LISTED",
        writeoff_status="STATUTORY",
    )

    result = calculate_risk_from_snapshot(
        policy=policy, snapshot=snapshot, policy_citations=citations
    )

    finding_map = {f.factor: f for f in result.findings}
    assert RiskFactor.MATCH in finding_map
    assert "obs-ppsr-001" in finding_map[RiskFactor.MATCH].evidence_refs
    assert "pass-ppsr-01" in finding_map[RiskFactor.MATCH].policy_citation_refs

    assert RiskFactor.LISTED in finding_map
    assert "obs-stolen-001" in finding_map[RiskFactor.LISTED].evidence_refs
    assert "pass-stolen-01" in finding_map[RiskFactor.LISTED].policy_citation_refs

    assert RiskFactor.STATUTORY in finding_map
    assert "obs-writeoff-001" in finding_map[RiskFactor.STATUTORY].evidence_refs
    assert "pass-statutory-01" in finding_map[RiskFactor.STATUTORY].policy_citation_refs


def test_incomplete_evidence_withholds_scoring() -> None:
    """When required evidence is incomplete or unverified, score and band MUST be None."""
    policy = build_risk_policy_v1()

    # 1. Unknown PPSR value
    snap_unknown_ppsr = _build_snapshot(ppsr_result="UNKNOWN")
    res_unknown = calculate_risk_from_snapshot(policy, snap_unknown_ppsr)
    assert res_unknown.is_incomplete is True
    assert res_unknown.outcome == AssessmentOutcome.INCOMPLETE
    assert res_unknown.score is None
    assert res_unknown.raw_score is None
    assert res_unknown.band is None
    assert len(res_unknown.missing_findings) >= 1
    assert any(m.field_name == "ppsr_result" for m in res_unknown.missing_findings)

    # 2. Unresolved Conflict
    conflict = FieldConflict(
        field_name="stolen_status",
        conflicting_candidates=(
            CandidateValue(
                field_name="stolen_status",
                value="LISTED",
                provenance=ProvenanceLink(
                    observation_id="obs-1",
                    source_system="POLICE",
                    source_record_id="pol-1",
                    retrieved_at=datetime.now(UTC),
                ),
            ),
            CandidateValue(
                field_name="stolen_status",
                value="NOT_STOLEN",
                provenance=ProvenanceLink(
                    observation_id="obs-2",
                    source_system="NZTA",
                    source_record_id="nzta-1",
                    retrieved_at=datetime.now(UTC),
                ),
            ),
        ),
        state=ConflictState.DETECTED,
    )
    snap_conflict = _build_snapshot(conflicts=(conflict,))
    res_conflict = calculate_risk_from_snapshot(policy, snap_conflict)
    assert res_conflict.is_incomplete is True
    assert res_conflict.score is None
    assert res_conflict.band is None
    assert any(
        m.field_name == "stolen_status" and m.reason == MissingEvidenceReason.UNRESOLVED_CONFLICT
        for m in res_conflict.missing_findings
    )


def test_unavailable_evidence_withholds_scoring() -> None:
    """When snapshot is None, calculation engine returns safe INCOMPLETE result."""
    policy = build_risk_policy_v1()
    result = calculate_risk_result(
        policy=policy,
        snapshot=None,
        assessment_id="asmt-none",
        run_number=1,
    )
    assert result.is_incomplete is True
    assert result.outcome == AssessmentOutcome.INCOMPLETE
    assert result.score is None
    assert result.band is None
    assert len(result.missing_findings) >= 1


def test_deterministic_result_hash() -> None:
    """Prove calculation hash is consistent across 50 runs and changes on input difference."""
    policy = build_risk_policy_v1()
    snapshot = _build_snapshot(ppsr_result="MATCH", stolen_status="NOT_STOLEN")

    hashes = set()
    for _ in range(50):
        res = calculate_risk_from_snapshot(policy, snapshot)
        hashes.add(res.calculation_hash)

    assert len(hashes) == 1
    orig_hash = hashes.pop()
    assert len(orig_hash) == 64

    # Different input yields different calculation_hash
    snap_diff = _build_snapshot(ppsr_result="NO_MATCH", stolen_status="NOT_STOLEN")
    res_diff = calculate_risk_from_snapshot(policy, snap_diff)
    assert res_diff.calculation_hash != orig_hash

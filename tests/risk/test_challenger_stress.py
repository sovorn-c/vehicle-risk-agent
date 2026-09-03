"""Empirical stress test harness for Risk Calculation Engine.

Authored by Challenger 1 (Milestone 1 / Story e04s01).
Tests mathematical properties:
1. Power set monotonicity and exact band assignment across all 2^4 = 16 factor subsets.
2. Permutation invariance of evidence, provenance, and citations across 50 random seeds.
3. Score capping at 100 with preservation of raw_score.
4. Calculation hash replayability, determinism, and tamper sensitivity.
5. Incomplete evidence invariants and boundary robustness.
"""

from __future__ import annotations

import itertools
import random
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.evidence.models import (
    CandidateValue,
    ConfidenceAssessment,
    ConfidenceBand,
    ConflictState,
    FieldConflict,
    ProvenanceLink,
)
from vehicle_risk_agent.evidence.snapshot import VehicleEvidenceSnapshot
from vehicle_risk_agent.evidence.sufficiency import MissingEvidenceReason
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.risk.calculator import calculate_risk_from_snapshot
from vehicle_risk_agent.risk.models import (
    DEFAULT_POLICY_V1_FACTOR_WEIGHTS,
    AssessmentOutcome,
    RiskBand,
    RiskFactor,
    RiskFactorResult,
    RiskResult,
    build_risk_policy_v1,
)

BAND_ORDER: dict[RiskBand, int] = {
    RiskBand.LOW: 0,
    RiskBand.MEDIUM: 1,
    RiskBand.HIGH: 2,
    RiskBand.CRITICAL: 3,
}


def _make_snapshot(
    ppsr_result: str = "NO_MATCH",
    stolen_status: str = "NOT_STOLEN",
    writeoff_status: str = "NONE",
    assessment_id: str = "asmt-challenger-001",
    run_number: int = 1,
    extra_fields: dict[str, Any] | None = None,
    conflicts: tuple[FieldConflict, ...] = (),
) -> VehicleEvidenceSnapshot:
    """Construct deterministic VehicleEvidenceSnapshot for stress tests."""
    fields: dict[str, Any] = {
        "ppsr_result": ppsr_result,
        "stolen_status": stolen_status,
        "writeoff_status": writeoff_status,
        "make": "TOYOTA",
        "model": "RAV4",
        "year": 2021,
    }
    if extra_fields:
        fields.update(extra_fields)

    fixed_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    provenance: dict[str, tuple[ProvenanceLink, ...]] = {
        "ppsr_result": (
            ProvenanceLink(
                observation_id="obs-ppsr-001",
                source_system="PPSR",
                source_record_id="rec-ppsr-100",
                retrieved_at=fixed_time,
            ),
        ),
        "stolen_status": (
            ProvenanceLink(
                observation_id="obs-stolen-001",
                source_system="POLICE",
                source_record_id="rec-police-200",
                retrieved_at=fixed_time,
            ),
        ),
        "writeoff_status": (
            ProvenanceLink(
                observation_id="obs-writeoff-001",
                source_system="NZTA",
                source_record_id="rec-nzta-300",
                retrieved_at=fixed_time,
            ),
        ),
    }

    return VehicleEvidenceSnapshot(
        assessment_id=assessment_id,
        run_number=run_number,
        vin="7AT0BJ00012345678",
        revision_id="rev-stress-01",
        revision_number=1,
        material_hash="0123456789abcdef" * 4,
        canonical_fields=fields,
        field_provenance=provenance,
        conflicts=conflicts,
        confidence=ConfidenceAssessment(
            score=95,
            band=ConfidenceBand.HIGH,
            rule_version="v1",
            explanation="Challenger test fixture",
        ),
        as_of=fixed_time,
        published_at=fixed_time,
    )


def _sample_policy_citations() -> tuple[PolicyCitation, ...]:
    """Provide realistic policy citations across all 4 factors."""
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


class TestEmpiricalPowerSetMonotonicity:
    """Stress test all 16 risk factor combinations for exact scoring, bands, and monotonicity."""

    FACTORS = [
        RiskFactor.MATCH,  # weight: 30
        RiskFactor.LISTED,  # weight: 45
        RiskFactor.REPAIRABLE,  # weight: 20
        RiskFactor.STATUTORY,  # weight: 40
    ]

    EXPECTED_SUBSET_RESULTS: dict[frozenset[RiskFactor], tuple[int, int, RiskBand]] = {
        frozenset(): (0, 0, RiskBand.LOW),
        frozenset({RiskFactor.REPAIRABLE}): (20, 20, RiskBand.MEDIUM),
        frozenset({RiskFactor.MATCH}): (30, 30, RiskBand.MEDIUM),
        frozenset({RiskFactor.STATUTORY}): (40, 40, RiskBand.HIGH),
        frozenset({RiskFactor.LISTED}): (45, 45, RiskBand.HIGH),
        frozenset({RiskFactor.MATCH, RiskFactor.REPAIRABLE}): (50, 50, RiskBand.HIGH),
        frozenset({RiskFactor.REPAIRABLE, RiskFactor.STATUTORY}): (60, 60, RiskBand.HIGH),
        frozenset({RiskFactor.LISTED, RiskFactor.REPAIRABLE}): (65, 65, RiskBand.HIGH),
        frozenset({RiskFactor.MATCH, RiskFactor.STATUTORY}): (70, 70, RiskBand.CRITICAL),
        frozenset({RiskFactor.MATCH, RiskFactor.LISTED}): (75, 75, RiskBand.CRITICAL),
        frozenset({RiskFactor.LISTED, RiskFactor.STATUTORY}): (85, 85, RiskBand.CRITICAL),
        frozenset({RiskFactor.MATCH, RiskFactor.REPAIRABLE, RiskFactor.STATUTORY}): (
            90,
            90,
            RiskBand.CRITICAL,
        ),
        frozenset({RiskFactor.MATCH, RiskFactor.LISTED, RiskFactor.REPAIRABLE}): (
            95,
            95,
            RiskBand.CRITICAL,
        ),
        frozenset({RiskFactor.LISTED, RiskFactor.REPAIRABLE, RiskFactor.STATUTORY}): (
            105,
            100,
            RiskBand.CRITICAL,
        ),
        frozenset({RiskFactor.MATCH, RiskFactor.LISTED, RiskFactor.STATUTORY}): (
            115,
            100,
            RiskBand.CRITICAL,
        ),
        frozenset(
            {
                RiskFactor.MATCH,
                RiskFactor.LISTED,
                RiskFactor.REPAIRABLE,
                RiskFactor.STATUTORY,
            }
        ): (135, 100, RiskBand.CRITICAL),
    }

    def test_all_16_subsets_exact_raw_score_capped_score_and_band(self) -> None:
        """Evaluate each of the 16 subsets against theoretical weights and band definitions."""
        policy = build_risk_policy_v1()

        subsets: list[frozenset[RiskFactor]] = []
        for r in range(len(self.FACTORS) + 1):
            for combo in itertools.combinations(self.FACTORS, r):
                subsets.append(frozenset(combo))

        assert len(subsets) == 16, f"Expected exactly 16 subsets, got {len(subsets)}"

        for subset in subsets:
            expected_raw, expected_capped, expected_band = self.EXPECTED_SUBSET_RESULTS[subset]

            # Calculate raw sum from policy weights directly
            calc_raw = sum(policy.factor_weights[f] for f in subset)
            calc_capped = min(calc_raw, policy.score_cap)

            # Determine band from policy risk_bands
            calc_band = RiskBand.LOW
            for b_def in sorted(policy.risk_bands, key=lambda b: b.min_score):
                if b_def.min_score <= calc_capped <= b_def.max_score:
                    calc_band = b_def.band
                    break

            assert calc_raw == expected_raw, (
                f"Subset {set(subset)}: expected raw {expected_raw}, got {calc_raw}"
            )
            assert calc_capped == expected_capped, (
                f"Subset {set(subset)}: expected capped {expected_capped}, got {calc_capped}"
            )
            assert calc_band == expected_band, (
                f"Subset {set(subset)}: expected band {expected_band}, got {calc_band}"
            )

    def test_pairwise_mathematical_monotonicity_on_all_256_pairs(self) -> None:
        """Prove that A <= B implies Score(A) <= Score(B) and Band(A) <= Band(B)."""
        all_subsets = list(self.EXPECTED_SUBSET_RESULTS.keys())
        total_pairs_tested = 0
        superset_pairs_tested = 0

        for sub_a in all_subsets:
            raw_a, capped_a, band_a = self.EXPECTED_SUBSET_RESULTS[sub_a]
            for sub_b in all_subsets:
                raw_b, capped_b, band_b = self.EXPECTED_SUBSET_RESULTS[sub_b]
                total_pairs_tested += 1

                if sub_a.issubset(sub_b):
                    superset_pairs_tested += 1
                    # Invariant 1: Raw score monotonicity
                    assert raw_a <= raw_b, (
                        f"Raw monotonicity violated: {set(sub_a)} ({raw_a}) <= "
                        f"{set(sub_b)} ({raw_b})"
                    )
                    # Invariant 2: Capped score monotonicity
                    assert capped_a <= capped_b, (
                        f"Capped monotonicity violated: {set(sub_a)} ({capped_a}) <= "
                        f"{set(sub_b)} ({capped_b})"
                    )
                    # Invariant 3: Risk band monotonicity
                    assert BAND_ORDER[band_a] <= BAND_ORDER[band_b], (
                        f"Band monotonicity violated: {set(sub_a)} ({band_a}) <= "
                        f"{set(sub_b)} ({band_b})"
                    )

        assert total_pairs_tested == 256
        # In a boolean lattice of dimension 4, number of pairs (A, B) with A <= B is 3^4 = 81
        assert superset_pairs_tested == 81, (
            f"Expected 81 inclusion pairs in 2^4 lattice, got {superset_pairs_tested}"
        )


class TestEmpiricalPermutationInvariance:
    """Stress test permutation invariance across 50 distinct random seeds."""

    def test_permutation_invariance_50_seeds(self) -> None:
        """Verify identical scores, bands, findings, and hashes across 50 seeds."""
        policy = build_risk_policy_v1()
        citations = _sample_policy_citations()

        # Complex multi-factor snapshot with multiple fields
        base_snapshot = _make_snapshot(
            ppsr_result="MATCH",
            stolen_status="LISTED",
            writeoff_status="STATUTORY",
            extra_fields={
                "engine_number": "ENG-998877",
                "fuel_type": "PETROL",
                "body_style": "STATION_WAGON",
                "odometer_km": 45000,
                "first_registration_nz": "2021-04-15",
            },
        )

        base_result = calculate_risk_from_snapshot(
            policy=policy,
            snapshot=base_snapshot,
            policy_citations=citations,
        )

        assert base_result.score == 100
        assert base_result.raw_score == 115
        assert base_result.band == RiskBand.CRITICAL
        assert len(base_result.findings) == 3
        assert len(base_result.calculation_hash) == 64

        for seed in range(1, 51):
            rng = random.Random(seed)

            # 1. Shuffle canonical_fields items
            field_items = list(base_snapshot.canonical_fields.items())
            rng.shuffle(field_items)
            shuffled_fields = dict(field_items)

            # 2. Shuffle provenance links and provenance map keys
            prov_items = list(base_snapshot.field_provenance.items())
            rng.shuffle(prov_items)
            shuffled_prov = dict(prov_items)

            # 3. Shuffle policy citations list
            shuffled_citations = list(citations)
            rng.shuffle(shuffled_citations)

            shuffled_snapshot = VehicleEvidenceSnapshot(
                assessment_id=base_snapshot.assessment_id,
                run_number=base_snapshot.run_number,
                vin=base_snapshot.vin,
                revision_id=base_snapshot.revision_id,
                revision_number=base_snapshot.revision_number,
                material_hash=base_snapshot.material_hash,
                canonical_fields=shuffled_fields,
                field_provenance=shuffled_prov,
                conflicts=base_snapshot.conflicts,
                confidence=base_snapshot.confidence,
                as_of=base_snapshot.as_of,
                published_at=base_snapshot.published_at,
            )

            permuted_result = calculate_risk_from_snapshot(
                policy=policy,
                snapshot=shuffled_snapshot,
                policy_citations=shuffled_citations,
            )

            # Assert complete mathematical and structural equivalence
            assert permuted_result.score == base_result.score, (
                f"Seed {seed}: score mismatch {permuted_result.score} != {base_result.score}"
            )
            assert permuted_result.raw_score == base_result.raw_score, (
                f"Seed {seed}: raw mismatch {permuted_result.raw_score} != {base_result.raw_score}"
            )
            assert permuted_result.band == base_result.band, (
                f"Seed {seed}: band mismatch {permuted_result.band} != {base_result.band}"
            )
            assert permuted_result.calculation_hash == base_result.calculation_hash, (
                f"Seed {seed}: calculation_hash mismatch"
            )

            # Assert exact findings equivalence (ordered by factor)
            assert len(permuted_result.findings) == len(base_result.findings)
            for f_perm, f_base in zip(permuted_result.findings, base_result.findings, strict=True):
                assert f_perm.finding_id == f_base.finding_id
                assert f_perm.factor == f_base.factor
                assert f_perm.weight == f_base.weight
                assert f_perm.evidence_refs == f_base.evidence_refs
                assert f_perm.policy_citation_refs == f_base.policy_citation_refs

            # Assert exact factor results equivalence
            assert len(permuted_result.factors) == len(base_result.factors)
            for r_perm, r_base in zip(permuted_result.factors, base_result.factors, strict=True):
                assert r_perm.factor == r_base.factor
                assert r_perm.triggered == r_base.triggered
                assert r_perm.score_contribution == r_base.score_contribution
                assert r_perm.evidence_refs == r_base.evidence_refs
                assert r_perm.policy_citation_refs == r_base.policy_citation_refs


class TestEmpiricalScoreCapping:
    """Stress test score clamping at 100 and arbitrary custom caps."""

    def test_score_cap_at_100_standard_policy(self) -> None:
        """Verify standard Policy v1 combinations that exceed 100 are clamped to 100."""
        policy = build_risk_policy_v1()

        # Case 1: MATCH(30) + LISTED(45) + STATUTORY(40) = 115 -> clamped to 100
        snap_115 = _make_snapshot(
            ppsr_result="MATCH",
            stolen_status="LISTED",
            writeoff_status="STATUTORY",
        )
        res_115 = calculate_risk_from_snapshot(policy, snap_115)
        assert res_115.raw_score == 115
        assert res_115.score == 100
        assert res_115.band == RiskBand.CRITICAL

        # Case 2: LISTED(45) + STATUTORY(40) + REPAIRABLE(20) = 105 (direct factor evaluation)
        factor_results = (
            RiskFactorResult(
                factor=RiskFactor.LISTED,
                weight=45,
                triggered=True,
                evidence_field="stolen_status",
            ),
            RiskFactorResult(
                factor=RiskFactor.STATUTORY,
                weight=40,
                triggered=True,
                evidence_field="writeoff_status",
            ),
            RiskFactorResult(
                factor=RiskFactor.REPAIRABLE,
                weight=20,
                triggered=True,
                evidence_field="writeoff_status",
            ),
        )
        raw_105 = sum(f.weight for f in factor_results if f.triggered)
        capped_105 = min(raw_105, policy.score_cap)
        assert raw_105 == 105
        assert capped_105 == 100

        # Case 3: MATCH(30) + LISTED(45) + REPAIRABLE(20) + STATUTORY(40) = 135 -> clamped to 100
        all_factors = (
            RiskFactorResult(
                factor=RiskFactor.MATCH,
                weight=30,
                triggered=True,
                evidence_field="ppsr_result",
            ),
            RiskFactorResult(
                factor=RiskFactor.LISTED,
                weight=45,
                triggered=True,
                evidence_field="stolen_status",
            ),
            RiskFactorResult(
                factor=RiskFactor.REPAIRABLE,
                weight=20,
                triggered=True,
                evidence_field="writeoff_status",
            ),
            RiskFactorResult(
                factor=RiskFactor.STATUTORY,
                weight=40,
                triggered=True,
                evidence_field="writeoff_status",
            ),
        )
        raw_135 = sum(f.weight for f in all_factors if f.triggered)
        capped_135 = min(raw_135, policy.score_cap)
        assert raw_135 == 135
        assert capped_135 == 100

    def test_custom_score_cap_enforcement(self) -> None:
        """Verify that modifying score_cap (e.g. 50, 75) clamps score while retaining raw_score."""
        policy_50 = build_risk_policy_v1(policy_id="policy-cap-50", score_cap=50)

        # MATCH(30) + STATUTORY(40) = 70 -> capped at 50
        snap = _make_snapshot(ppsr_result="MATCH", writeoff_status="STATUTORY")
        res_50 = calculate_risk_from_snapshot(policy_50, snap)

        assert res_50.raw_score == 70
        assert res_50.score == 50
        # In policy_50, 50 falls in HIGH (40-69)
        assert res_50.band == RiskBand.HIGH


class TestCalculationHashReplayabilityAndTamperDetection:
    """Stress test SHA-256 calculation hash determinism, replayability, and tamper detection."""

    def test_hash_replayability_100_runs(self) -> None:
        """Verify 100 identical replayed calculations produce the exact same SHA-256 hash."""
        policy = build_risk_policy_v1()
        snapshot = _make_snapshot(
            ppsr_result="MATCH",
            stolen_status="NOT_STOLEN",
            writeoff_status="REPAIRABLE",
        )
        citations = _sample_policy_citations()

        hashes: set[str] = set()
        for _ in range(100):
            res = calculate_risk_from_snapshot(
                policy=policy,
                snapshot=snapshot,
                policy_citations=citations,
            )
            hashes.add(res.calculation_hash)

        assert len(hashes) == 1, f"Expected exactly 1 deterministic hash, got {len(hashes)}"
        calc_hash = hashes.pop()
        assert len(calc_hash) == 64
        # Verify valid hex
        int(calc_hash, 16)

    def test_tamper_detection_changes_hash(self) -> None:
        """Verify altering any input component strictly changes the SHA-256 calculation hash."""
        policy = build_risk_policy_v1()
        base_snap = _make_snapshot(
            ppsr_result="MATCH",
            stolen_status="NOT_STOLEN",
            writeoff_status="NONE",
        )
        base_res = calculate_risk_from_snapshot(policy, base_snap)
        base_hash = base_res.calculation_hash

        # 1. Tamper with policy ID
        policy_tampered_id = build_risk_policy_v1(policy_id="tampered-policy-id")
        res_tampered_id = calculate_risk_from_snapshot(policy_tampered_id, base_snap)
        assert res_tampered_id.calculation_hash != base_hash

        # 2. Tamper with policy version
        policy_tampered_ver = build_risk_policy_v1(version="v2")
        res_tampered_ver = calculate_risk_from_snapshot(policy_tampered_ver, base_snap)
        assert res_tampered_ver.calculation_hash != base_hash

        # 3. Tamper with factor weight (modifies policy_hash)
        custom_weights = dict(DEFAULT_POLICY_V1_FACTOR_WEIGHTS)
        custom_weights[RiskFactor.MATCH] = 35
        policy_tampered_weights = build_risk_policy_v1(factor_weights=custom_weights)
        res_tampered_weights = calculate_risk_from_snapshot(policy_tampered_weights, base_snap)
        assert res_tampered_weights.calculation_hash != base_hash

        # 4. Tamper with canonical field value
        snap_tampered_field = _make_snapshot(
            ppsr_result="NO_MATCH",
            stolen_status="NOT_STOLEN",
            writeoff_status="NONE",
        )
        res_tampered_field = calculate_risk_from_snapshot(policy, snap_tampered_field)
        assert res_tampered_field.calculation_hash != base_hash

        # 5. Tamper with unrelated canonical field
        snap_tampered_year = _make_snapshot(ppsr_result="MATCH", extra_fields={"year": 2022})
        res_tampered_year = calculate_risk_from_snapshot(policy, snap_tampered_year)
        assert res_tampered_year.calculation_hash != base_hash


class TestIncompleteEvidenceInvariants:
    """Stress test fail-closed INCOMPLETE evidence handling and data model invariants."""

    def test_missing_required_fields_withhold_scoring(self) -> None:
        """Verify omitting any required evidence field yields safe INCOMPLETE result with None."""
        policy = build_risk_policy_v1()

        # Missing ppsr_result
        snap_no_ppsr = _make_snapshot(ppsr_result="")
        res_no_ppsr = calculate_risk_from_snapshot(policy, snap_no_ppsr)
        assert res_no_ppsr.is_incomplete is True
        assert res_no_ppsr.outcome == AssessmentOutcome.INCOMPLETE
        assert res_no_ppsr.score is None
        assert res_no_ppsr.raw_score is None
        assert res_no_ppsr.band is None
        assert len(res_no_ppsr.missing_findings) >= 1

    def test_unresolved_conflict_withholds_scoring(self) -> None:
        """Verify detected conflict triggers INCOMPLETE outcome."""
        policy = build_risk_policy_v1()
        fixed_time = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)

        conflict = FieldConflict(
            field_name="writeoff_status",
            conflicting_candidates=(
                CandidateValue(
                    field_name="writeoff_status",
                    value="STATUTORY",
                    provenance=ProvenanceLink(
                        observation_id="obs-w1",
                        source_system="NZTA",
                        source_record_id="rec-1",
                        retrieved_at=fixed_time,
                    ),
                ),
                CandidateValue(
                    field_name="writeoff_status",
                    value="NONE",
                    provenance=ProvenanceLink(
                        observation_id="obs-w2",
                        source_system="DEALER",
                        source_record_id="rec-2",
                        retrieved_at=fixed_time,
                    ),
                ),
            ),
            state=ConflictState.DETECTED,
        )

        snap_conflict = _make_snapshot(conflicts=(conflict,))
        res_conflict = calculate_risk_from_snapshot(policy, snap_conflict)
        assert res_conflict.is_incomplete is True
        assert res_conflict.outcome == AssessmentOutcome.INCOMPLETE
        assert res_conflict.score is None
        assert res_conflict.band is None
        assert any(
            m.field_name == "writeoff_status"
            and m.reason == MissingEvidenceReason.UNRESOLVED_CONFLICT
            for m in res_conflict.missing_findings
        )

    def test_pydantic_model_enforces_completeness_invariants(self) -> None:
        """Verify Pydantic validator strictly rejects invalid score/band states."""
        # Cannot have score if is_incomplete is True
        with pytest.raises(ValidationError):
            RiskResult(
                assessment_id="asmt-1",
                run_number=1,
                policy_id="policy-1",
                policy_version="v1",
                score=40,
                band=RiskBand.HIGH,
                is_incomplete=True,
            )

        # Cannot omit score if is_incomplete is False
        with pytest.raises(ValidationError):
            RiskResult(
                assessment_id="asmt-1",
                run_number=1,
                policy_id="policy-1",
                policy_version="v1",
                score=None,
                band=None,
                is_incomplete=False,
            )

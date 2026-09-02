"""Unit tests for Risk Policy domain models, immutability, and invariants."""

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.risk.models import (
    MandatoryFinding,
    RiskBand,
    RiskBandDefinition,
    RiskFactor,
    RiskFactorResult,
    RiskPolicy,
    RiskPolicyLifecycleState,
    RiskResult,
    build_risk_policy_v1,
)


def test_risk_policy_v1_defaults() -> None:
    """Verify that build_risk_policy_v1 instantiates standard Policy v1."""
    policy = build_risk_policy_v1()

    assert policy.id == "risk-policy-v1"
    assert policy.version == "v1"
    assert policy.lifecycle_state == RiskPolicyLifecycleState.DRAFT
    assert policy.score_cap == 100

    # Factor weights
    assert policy.factor_weights[RiskFactor.MATCH] == 30
    assert policy.factor_weights[RiskFactor.LISTED] == 45
    assert policy.factor_weights[RiskFactor.REPAIRABLE] == 20
    assert policy.factor_weights[RiskFactor.STATUTORY] == 40

    # Risk bands
    band_map = {b.band: (b.min_score, b.max_score) for b in policy.risk_bands}
    assert band_map[RiskBand.LOW] == (0, 19)
    assert band_map[RiskBand.MEDIUM] == (20, 39)
    assert band_map[RiskBand.HIGH] == (40, 69)
    assert band_map[RiskBand.CRITICAL] == (70, 100)

    # Required evidence
    assert policy.required_evidence_fields == ("ppsr_result", "stolen_status", "writeoff_status")
    assert policy.policy_hash is not None
    assert len(policy.policy_hash) == 64


def test_risk_policy_immutability() -> None:
    """Verify frozen immutability on RiskPolicy and related models."""
    policy = build_risk_policy_v1()

    with pytest.raises((ValidationError, TypeError)):
        policy.score_cap = 90

    with pytest.raises((ValidationError, TypeError)):
        policy.lifecycle_state = RiskPolicyLifecycleState.ACTIVE

    finding = MandatoryFinding(
        finding_id="finding-test-1",
        factor=RiskFactor.MATCH,
        title="Test Match Finding",
        description="PPSR security interest detected",
        evidence_field="ppsr_result",
        evidence_value="MATCH",
        weight=30,
    )
    with pytest.raises((ValidationError, TypeError)):
        finding.weight = 50

    factor_result = RiskFactorResult(
        factor=RiskFactor.MATCH,
        weight=30,
        triggered=True,
        evidence_field="ppsr_result",
        evidence_value="MATCH",
        rationale="Match found on PPSR",
    )
    with pytest.raises((ValidationError, TypeError)):
        factor_result.triggered = False


def test_risk_policy_deterministic_hash() -> None:
    """Verify policy_hash is deterministic and changes with any parameter alteration."""
    policy1 = build_risk_policy_v1(policy_id="policy-1")
    policy2 = build_risk_policy_v1(policy_id="policy-1")

    assert policy1.policy_hash == policy2.policy_hash
    assert len(policy1.policy_hash) == 64

    # Altering weight changes hash
    modified_weights = dict(policy1.factor_weights)
    modified_weights[RiskFactor.MATCH] = 35
    policy_mod_weights = build_risk_policy_v1(policy_id="policy-1", factor_weights=modified_weights)
    assert policy_mod_weights.policy_hash != policy1.policy_hash

    # Altering score cap changes hash
    policy_mod_cap = build_risk_policy_v1(policy_id="policy-1", score_cap=90)
    assert policy_mod_cap.policy_hash != policy1.policy_hash


def test_risk_policy_validation_rejects_invalid_weights() -> None:
    """Verify that negative weights or weights exceeding 100 are rejected."""
    with pytest.raises(ValidationError):
        RiskPolicy(
            id="invalid-weights-1",
            version="v1",
            name="Invalid Policy",
            description="Testing invalid weights",
            factor_weights={
                RiskFactor.MATCH: -10,
                RiskFactor.LISTED: 45,
                RiskFactor.REPAIRABLE: 20,
                RiskFactor.STATUTORY: 40,
            },
        )

    with pytest.raises(ValidationError):
        RiskPolicy(
            id="invalid-weights-2",
            version="v1",
            name="Invalid Policy",
            description="Testing invalid weights",
            factor_weights={
                RiskFactor.MATCH: 120,
                RiskFactor.LISTED: 45,
                RiskFactor.REPAIRABLE: 20,
                RiskFactor.STATUTORY: 40,
            },
        )


def test_risk_policy_validation_rejects_invalid_band_thresholds() -> None:
    """Verify that overlapping or inverted score intervals are rejected."""
    # Inverted interval (min > max)
    with pytest.raises(ValidationError):
        RiskBandDefinition(
            band=RiskBand.LOW,
            min_score=30,
            max_score=10,
            description="Invalid band interval",
        )

    # Gaps or overlapping bands in policy
    with pytest.raises(ValidationError):
        RiskPolicy(
            id="invalid-bands-1",
            version="v1",
            name="Invalid Policy",
            description="Testing invalid bands",
            risk_bands=(
                RiskBandDefinition(band=RiskBand.LOW, min_score=0, max_score=25, description="Low"),
                RiskBandDefinition(
                    band=RiskBand.MEDIUM, min_score=20, max_score=40, description="Medium"
                ),  # Overlap with Low
                RiskBandDefinition(
                    band=RiskBand.HIGH, min_score=41, max_score=69, description="High"
                ),
                RiskBandDefinition(
                    band=RiskBand.CRITICAL, min_score=70, max_score=100, description="Critical"
                ),
            ),
        )


def test_mandatory_finding_schema_and_immutability() -> None:
    """Validate MandatoryFinding construction, field preservation, and frozen attributes."""
    finding = MandatoryFinding(
        finding_id="finding-stolen-123",
        factor=RiskFactor.LISTED,
        title="Active Stolen Vehicle Record",
        description="Police register indicates the vehicle is listed as stolen.",
        evidence_field="stolen_status",
        evidence_value="LISTED",
        weight=45,
        evidence_refs=("obs-001", "obs-002"),
        policy_citation_refs=("passage-001",),
    )

    assert finding.finding_id == "finding-stolen-123"
    assert finding.factor == RiskFactor.LISTED
    assert finding.weight == 45
    assert finding.evidence_refs == ("obs-001", "obs-002")
    assert finding.policy_citation_refs == ("passage-001",)


def test_risk_factor_result_schema_and_immutability() -> None:
    """Validate RiskFactorResult construction and fields."""
    result = RiskFactorResult(
        factor=RiskFactor.STATUTORY,
        weight=40,
        triggered=True,
        evidence_field="writeoff_status",
        evidence_value="STATUTORY",
        rationale="Statutory write-off detected on register",
        evidence_refs=("obs-statutory-1",),
        policy_citation_refs=("citation-statutory-1",),
    )

    assert result.factor == RiskFactor.STATUTORY
    assert result.weight == 40
    assert result.triggered is True
    assert result.evidence_field == "writeoff_status"
    assert result.evidence_value == "STATUTORY"
    assert result.evidence_refs == ("obs-statutory-1",)
    assert result.policy_citation_refs == ("citation-statutory-1",)


def test_risk_result_invariants() -> None:
    """Verify completeness invariants on RiskResult."""
    # Complete result requires score and band
    with pytest.raises(ValidationError):
        RiskResult(
            assessment_id="ast-001",
            run_number=1,
            policy_id="risk-policy-v1",
            policy_version="v1",
            score=None,
            band=None,
            is_incomplete=False,
        )

    # Incomplete result forbids score and band
    with pytest.raises(ValidationError):
        RiskResult(
            assessment_id="ast-001",
            run_number=1,
            policy_id="risk-policy-v1",
            policy_version="v1",
            score=50,
            band=RiskBand.HIGH,
            is_incomplete=True,
        )

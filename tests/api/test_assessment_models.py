"""Tests for Assessment input models and strict validation contracts."""

import pytest
from pydantic import ValidationError

from vehicle_risk_agent.api.models import AssessmentContext, AssessmentCreateRequest, SaleType


def test_valid_assessment_create_request() -> None:
    """Verify well-formed assessment request parses successfully."""
    req = AssessmentCreateRequest(
        vin="1HGCR2F85HA000000",
        context=AssessmentContext(
            sale_type=SaleType.DEALER,
            intended_use="Daily commuter in Auckland",
            questions=["Has this vehicle been reported stolen?"],
        ),
    )
    assert req.vin == "1HGCR2F85HA000000"
    assert req.context.sale_type == SaleType.DEALER
    assert len(req.context.questions) == 1


def test_invalid_vin_length_rejected() -> None:
    """Verify VIN must be exactly 17 characters."""
    with pytest.raises(ValidationError) as exc:
        AssessmentCreateRequest(
            vin="SHORT123",
            context=AssessmentContext(sale_type=SaleType.PRIVATE),
        )
    assert "17 characters" in str(exc.value)


def test_invalid_vin_characters_rejected() -> None:
    """Verify letters I, O, Q are disallowed per ISO 3779 standard."""
    with pytest.raises(ValidationError) as exc:
        AssessmentCreateRequest(
            vin="1HGCR2F85HA00000I",
            context=AssessmentContext(sale_type=SaleType.PRIVATE),
        )
    assert "disallowed letters I, O, Q" in str(exc.value)


def test_invalid_vin_check_digit_rejected() -> None:
    """Verify check digit verification failure raises validation error."""
    # 1HGCR2F85HA000000 has valid check digit 5. Corrupt position 9 to 7.
    with pytest.raises(ValidationError) as exc:
        AssessmentCreateRequest(
            vin="1HGCR2F87HA000000",
            context=AssessmentContext(sale_type=SaleType.PRIVATE),
        )
    assert "Invalid VIN check digit" in str(exc.value)


def test_context_bounded_questions_cardinality() -> None:
    """Verify maximum of five questions in Assessment Context."""
    with pytest.raises(ValidationError) as exc:
        AssessmentContext(
            sale_type=SaleType.PRIVATE,
            questions=[f"Question {i}" for i in range(6)],
        )
    assert "at most 5 questions" in str(exc.value)


def test_context_bounded_question_length() -> None:
    """Verify maximum 200 characters per question in Assessment Context."""
    with pytest.raises(ValidationError) as exc:
        AssessmentContext(
            sale_type=SaleType.PRIVATE,
            questions=["A" * 201],
        )
    assert "at most 200 characters" in str(exc.value)


def test_request_rejects_extra_fields() -> None:
    """Verify strict request model rejects unknown extra fields."""
    with pytest.raises(ValidationError):
        AssessmentCreateRequest.model_validate(
            {
                "vin": "1HGCR2F85HA000000",
                "context": {"sale_type": "PRIVATE"},
                "extra_malicious_field": "injected",
            }
        )

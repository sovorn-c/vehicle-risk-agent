"""ISO 3779 VIN validator and check digit calculation."""

from dataclasses import dataclass

VIN_TRANSLITERATION: dict[str, int] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "J": 1,
    "K": 2,
    "L": 3,
    "M": 4,
    "N": 5,
    "P": 7,
    "R": 9,
    "S": 2,
    "T": 3,
    "U": 4,
    "V": 5,
    "W": 6,
    "X": 7,
    "Y": 8,
    "Z": 9,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
}

VIN_POSITION_WEIGHTS: list[int] = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
FORBIDDEN_VIN_CHARS: set[str] = {"I", "O", "Q"}


@dataclass(frozen=True)
class VinValidationResult:
    """Result of VIN validation."""

    is_valid: bool
    normalized_vin: str | None = None
    error_reason: str | None = None


def calculate_vin_check_digit(vin: str) -> str | None:
    """Calculate the ISO 3779 check digit for a 17-character VIN."""
    if len(vin) != 17:
        return None

    total = 0
    for i, char in enumerate(vin.upper()):
        if char not in VIN_TRANSLITERATION:
            return None
        total += VIN_TRANSLITERATION[char] * VIN_POSITION_WEIGHTS[i]

    remainder = total % 11
    if remainder == 10:
        return "X"
    return str(remainder)


def validate_vin(raw_vin: str) -> VinValidationResult:
    """Validate a VIN against ISO 3779 requirements."""
    if not raw_vin:
        return VinValidationResult(is_valid=False, error_reason="VIN cannot be empty")

    normalized = raw_vin.strip().upper()

    if len(normalized) != 17:
        return VinValidationResult(
            is_valid=False,
            error_reason=f"VIN must be exactly 17 characters (got {len(normalized)})",
        )

    for forbidden in FORBIDDEN_VIN_CHARS:
        if forbidden in normalized:
            return VinValidationResult(
                is_valid=False,
                error_reason=f"VIN contains disallowed letters I, O, Q: '{forbidden}'",
            )

    expected_check_digit = calculate_vin_check_digit(normalized)
    if expected_check_digit is None:
        return VinValidationResult(
            is_valid=False,
            error_reason="VIN contains invalid characters for transliteration",
        )

    actual_check_digit = normalized[8]
    if actual_check_digit != expected_check_digit:
        return VinValidationResult(
            is_valid=False,
            error_reason=(
                f"Invalid VIN check digit at position 9: "
                f"expected '{expected_check_digit}', got '{actual_check_digit}'"
            ),
        )

    return VinValidationResult(is_valid=True, normalized_vin=normalized)

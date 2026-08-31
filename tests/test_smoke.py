"""Baseline smoke test for package setup."""

from vehicle_risk_agent import __version__


def test_package_version() -> None:
    """Verify package exposes a semantic version."""
    assert __version__ == "0.1.0"

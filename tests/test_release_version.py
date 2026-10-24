"""Release gate: a tag must agree with the packaged project version."""

import subprocess
import sys

import pytest

from vehicle_risk_agent import __version__

SCRIPT = "scripts/check_release_version.py"


def run_gate(tag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, SCRIPT, tag],
        capture_output=True,
        text=True,
        check=False,
    )


def test_matching_tag_passes() -> None:
    result = run_gate(f"v{__version__}")
    assert result.returncode == 0
    assert f"release version OK: {__version__}" in result.stdout


def test_tag_not_matching_project_version_fails() -> None:
    result = run_gate("v9.9.9")
    assert result.returncode == 1
    assert result.stderr.startswith("ERROR:")
    assert "9.9.9" in result.stderr
    assert __version__ in result.stderr


@pytest.mark.parametrize("tag", ["0.4.0", "v0.4", "v0.4.0-rc1", ""])
def test_malformed_tags_fail(tag: str) -> None:
    result = run_gate(tag)
    assert result.returncode == 1
    assert result.stderr.startswith("ERROR:")

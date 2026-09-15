"""Acceptance contracts for the public e12 measurement walkthrough."""

import json
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_walkthrough_requires_real_boundaries_and_stops_before_review() -> None:
    script = (ROOT / "scripts/demo-measured-quality.sh").read_text(encoding="utf-8")

    assert "GEMINI_API_KEY" in script
    assert "E12_PROVIDER" in script
    assert "MCP_SERVER_URL" in script
    assert "--enable-live-eval" in script
    assert "--suite e12-comparative" in script
    assert "AWAITING_REVIEW" in script
    assert "APPROVE_REPORT" not in script
    assert "BLOCKED:" in script
    assert "evaluation.publication" in script
    assert "synthetic" in script.lower()


def test_public_copy_publishes_only_the_current_blocked_control_state() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "src/vehicle_risk_agent/api/docs.py").read_text(encoding="utf-8")
    publication = json.loads(
        (ROOT / "artifacts/e12-comparative-report.publication.json").read_text(encoding="utf-8")
    )

    for document in (readme, docs):
        assert "benchmark" in document.lower() or "comparative" in document.lower()
        assert publication["release_verdict"] in document
        assert "PostgreSQL full-text search" in document
        assert "live model validation" in document.lower()
        assert publication["config_hash"] in document
        for binding in (
            "source_commit",
            "config_sha256",
            "judgments_sha256",
            "evaluation_input_sha256",
            "report_bytes_sha256",
            "report_hash",
        ):
            assert publication[binding] in document
        assert "e471c0c687736cac51316dc7d91ccaf78f738cc01b03eea1a02a00f873da85fb" not in document
        assert "BM25" not in document

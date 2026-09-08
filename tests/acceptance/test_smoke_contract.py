"""Acceptance contracts for the real local-stack smoke entrypoint."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_local_smoke_starts_and_tears_down_compose_stack() -> None:
    """The operator script must own the full Compose lifecycle."""
    content = (REPO_ROOT / "scripts" / "smoke-local.sh").read_text(encoding="utf-8")
    assert "docker compose -f compose.yaml up" in content
    assert "docker compose -f compose.yaml down -v" in content
    assert "docker compose -f compose.yaml restart agent-api" in content
    assert "vehicle_risk_agent.cli.smoke" in content


def test_live_smoke_uses_http_api_and_not_fake_workflow_adapters() -> None:
    """Live smoke must enter through Agent API rather than bypassing service boundaries."""
    content = (REPO_ROOT / "src" / "vehicle_risk_agent" / "cli" / "smoke.py").read_text(
        encoding="utf-8"
    )
    assert "AsyncClient" in content
    assert "/api/v1/assessments" in content
    assert "FakeVehicleMcpAdapter" not in content
    assert "AssessmentWorkflowRunner" not in content

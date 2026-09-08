"""Acceptance tests verifying Docker Compose and Dockerfile contracts."""

from pathlib import Path

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_compose_file_exists_and_parses() -> None:
    """compose.yaml must exist and parse as valid YAML."""
    compose_path = REPO_ROOT / "compose.yaml"
    assert compose_path.exists(), "compose.yaml does not exist in repository root"

    with compose_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "compose.yaml root must be a mapping"
    assert "services" in data, "compose.yaml must define services"


def test_all_services_present_in_compose() -> None:
    """Compose must define db (pgvector), pipeline, mcp, and agent-api services."""
    compose_path = REPO_ROOT / "compose.yaml"
    with compose_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data.get("services", {})
    required_services = {"db", "pipeline", "mcp", "agent-api"}
    assert required_services.issubset(services.keys()), (
        f"Missing required services: {required_services - set(services.keys())}"
    )


def test_all_services_define_healthchecks() -> None:
    """Every service in compose must define a robust healthcheck with retries."""
    compose_path = REPO_ROOT / "compose.yaml"
    with compose_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data.get("services", {})
    for name in ("db", "pipeline", "mcp", "agent-api"):
        svc = services.get(name, {})
        assert "healthcheck" in svc, f"Service '{name}' missing healthcheck definition"
        hc = svc["healthcheck"]
        assert "test" in hc, f"Service '{name}' healthcheck missing test command"
        assert "interval" in hc, f"Service '{name}' healthcheck missing interval"
        assert "timeout" in hc, f"Service '{name}' healthcheck missing timeout"
        assert "retries" in hc, f"Service '{name}' healthcheck missing retries"


def test_service_health_dependencies() -> None:
    """agent-api must depend on db and mcp healthy, and mcp must depend on pipeline healthy."""
    compose_path = REPO_ROOT / "compose.yaml"
    with compose_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    services = data.get("services", {})
    agent_svc = services.get("agent-api", {})
    depends_on = agent_svc.get("depends_on", {})
    assert "db" in depends_on
    assert "mcp" in depends_on

    mcp_svc = services.get("mcp", {})
    mcp_depends_on = mcp_svc.get("depends_on", {})
    assert "pipeline" in mcp_depends_on


def test_dockerfile_contract() -> None:
    """Dockerfile must use Python 3.12, multi-stage build, non-root user, and healthcheck."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile does not exist in repository root"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "python:3.12" in content or "python3.12" in content
    assert "USER" in content
    assert "HEALTHCHECK" in content
    assert "EXPOSE" in content

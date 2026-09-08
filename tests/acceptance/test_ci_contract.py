"""Acceptance contract test for CI pipeline configuration."""

# story: e07s03

from pathlib import Path

import yaml  # type: ignore[import-untyped]


def test_ci_workflow_exists_and_valid() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    ci_path = repo_root / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists(), "Expected .github/workflows/ci.yml to exist"

    with open(ci_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert "name" in config
    assert "on" in config or True in config

    # Triggers must include push, pull_request, and workflow_dispatch for manual live evals
    triggers = config.get("on") if "on" in config else config.get(True)
    assert triggers is not None
    if isinstance(triggers, (list, dict)):
        assert "push" in triggers
        assert "pull_request" in triggers
        assert "workflow_dispatch" in triggers


def test_ci_deterministic_gates_no_paid_credentials() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    ci_path = repo_root / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists()

    with open(ci_path, encoding="utf-8") as f:
        content = f.read()
        config = yaml.safe_load(content)

    # Must not require paid API keys for default CI test run
    jobs = config.get("jobs", {})
    assert "test" in jobs or "preflight" in jobs

    test_job = jobs.get("test") or jobs.get("preflight")
    assert test_job is not None

    # Verify python 3.12 pinned
    assert "3.12" in content

    # Verify deterministic checks are present
    deterministic = "check.sh" in content or (
        "ruff" in content and "mypy" in content and "pytest" in content
    )
    assert deterministic

    # Verify pgvector service container is configured for database tests
    assert "pgvector" in content

    # Verify no hardcoded secrets or mandatory paid credentials in standard job env
    job_env = test_job.get("env", {})
    assert "ANTHROPIC_API_KEY" not in job_env or "${{" in str(job_env.get("ANTHROPIC_API_KEY"))
    assert "OPENAI_API_KEY" not in job_env or "${{" in str(job_env.get("OPENAI_API_KEY"))

    # If live eval job exists, it must be conditioned or manual
    if "live-eval" in jobs:
        live_job = jobs["live-eval"]
        # Must only run on workflow_dispatch or explicit condition
        assert "workflow_dispatch" in str(config.get("on")) or "if" in live_job


def test_ci_database_url_and_driver_contract() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    ci_path = repo_root / ".github" / "workflows" / "ci.yml"
    assert ci_path.exists()

    content = ci_path.read_text(encoding="utf-8")
    assert "DATABASE_URL:" in content or "DATABASE_URL" in content, "CI must use DATABASE_URL"
    assert "VEHICLE_RISK_AGENT_DATABASE_URL" not in content, (
        "CI must not use unused VEHICLE_RISK_AGENT_DATABASE_URL"
    )
    assert "asyncpg" not in content, "CI must not reference uninstalled asyncpg driver"
    assert "psycopg" in content, "CI must use psycopg driver matching pyproject.toml"


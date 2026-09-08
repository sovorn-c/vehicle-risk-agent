"""Acceptance contract test verifying documentation completeness and command integrity."""

# story: e07s03

from pathlib import Path


def test_readme_contains_required_sections() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    readme_path = repo_root / "README.md"
    assert readme_path.exists(), "README.md must exist"

    content = readme_path.read_text(encoding="utf-8")
    content_lower = content.lower()

    # Core required sections per e07s03-t03
    assert "architecture" in content_lower, "README must document architecture"
    assert "provenance" in content_lower or "policy" in content_lower
    assert "security" in content_lower, "README must document security boundaries"
    assert "limitation" in content_lower, "README must document operational limitations"
    assert "quickstart" in content_lower, "README must provide a quickstart guide"
    assert "scenario" in content_lower or "smoke" in content_lower
    assert "review" in content_lower, "README must document reviewer human-in-the-loop flow"
    assert "troubleshoot" in content_lower, "README must provide troubleshooting steps"
    assert "cleanup" in content_lower or "teardown" in content_lower


def test_readme_commands_and_contracts() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    readme_path = repo_root / "README.md"
    assert readme_path.exists()

    content = readme_path.read_text(encoding="utf-8")

    # Required executable commands documented in quickstart / verification
    required_commands = [
        "scripts/check.sh",
        "scripts/smoke-local.sh",
        "vehicle_risk_agent.cli.seed",
        "vehicle_risk_agent.cli.smoke",
        "compose.yaml",
    ]
    for cmd in required_commands:
        assert cmd in content, f"README must mention command or artifact: {cmd}"

    # Required domain boundary rules documented
    assert "deterministic" in content.lower(), "README must state deterministic scoring rules"
    assert "redaction" in content.lower() or "redact" in content.lower()
    assert "human" in content.lower(), "README must document human approval requirement"


def test_env_example_exists_and_matches_compose() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    env_example = repo_root / ".env.example"
    assert env_example.exists(), ".env.example must exist in repository root"
    content = env_example.read_text(encoding="utf-8")
    assert "DATABASE_URL" in content
    assert "psycopg" in content
    assert "54329" in content


def test_readme_ports_match_compose() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    readme_path = repo_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "54329" in content, "README must document PostgreSQL port 54329"
    assert "8001" in content, "README must document agent-api port 8001"
    assert "8080" in content, "README must document MCP port 8080"
    assert "asyncpg" not in content, "README must use psycopg, not asyncpg"

"""Acceptance contract for the tag-triggered release workflow and image build args."""

# story: e13s05

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RELEASE_PATH = REPO_ROOT / ".github" / "workflows" / "release.yml"
E12_EMBEDDING_REVISION = "8b3219a92973c328a8e22fadcfa821b5dc75636a"
E12_RERANKER_REVISION = "739bce82df32cacea8ff0edf73ab49ae315e5a5f"
PINNED_USES = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")


def _release() -> dict[Any, Any]:
    with RELEASE_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert isinstance(config, dict)
    return config


def _steps() -> list[dict[str, Any]]:
    steps = _release()["jobs"]["release"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step_index(predicate: str) -> int:
    for index, step in enumerate(_steps()):
        if predicate in str(step.get("run", "")) or predicate in str(step.get("uses", "")):
            return index
    raise AssertionError(f"no release step matches {predicate!r}")


def test_release_triggers_only_on_semver_tags() -> None:
    config = _release()
    triggers = config.get("on", config.get(True))
    assert triggers == {"push": {"tags": ["v*.*.*"]}}


def test_every_action_is_pinned_to_a_full_commit_sha() -> None:
    uses = [step["uses"] for step in _steps() if "uses" in step]
    assert uses
    for reference in uses:
        assert PINNED_USES.fullmatch(reference), f"unpinned action: {reference}"


def test_release_job_permissions_are_exactly_the_publish_set() -> None:
    assert _release()["permissions"] == {"contents": "read"}
    assert _release()["jobs"]["release"]["permissions"] == {
        "contents": "read",
        "packages": "write",
        "id-token": "write",
        "attestations": "write",
    }


def test_image_is_amd64_with_ml_and_azure_and_no_latest_tag() -> None:
    build = _steps()[_step_index("docker/build-push-action")]["with"]
    assert build["platforms"] == "linux/amd64"
    assert "INSTALL_ML=true" in build["build-args"]
    assert "INSTALL_AZURE=true" in build["build-args"]
    meta = _steps()[_step_index("docker/metadata-action")]["with"]
    assert meta["images"] == "ghcr.io/sovorn-c/vehicle-risk-agent"
    assert meta["flavor"] == "latest=false"
    assert "type=semver,pattern={{version}}" in meta["tags"]


def test_version_preflight_and_predeploy_gates_run_before_image_build() -> None:
    build = _step_index("docker/build-push-action")
    assert _step_index("scripts/check_release_version.py") < build
    assert _step_index("bash scripts/check.sh") < build
    assert _step_index("bash scripts/predeploy-eval.sh") < build


def test_dockerfile_declares_azure_extra_and_e12_model_revisions() -> None:
    content = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG INSTALL_AZURE=false" in content
    assert f"ARG EMBEDDING_REVISION={E12_EMBEDDING_REVISION}" in content
    assert f"ARG RERANKER_REVISION={E12_RERANKER_REVISION}" in content
    assert (
        content.count('if [ "$INSTALL_AZURE" = "true" ]; then extras="$extras --extra azure"') == 2
    )
    for runtime_env in (
        "HF_HUB_OFFLINE=1",
        "EMBEDDING_REVISION=$EMBEDDING_REVISION",
        "RERANKER_REVISION=$RERANKER_REVISION",
    ):
        assert runtime_env in content


def test_core_app_and_gate_import_without_azure_sdks() -> None:
    """The no-extra install path must not import the optional Azure SDKs."""
    probe = (
        "import sys, vehicle_risk_agent.api.app, vehicle_risk_agent.cli.predeploy_eval;"
        "print(sorted(m for m in ('openai', 'azure.identity') if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "[]"

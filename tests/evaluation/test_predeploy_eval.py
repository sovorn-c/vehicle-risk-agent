"""Contract tests for the offline predeploy gate over the frozen e12 config."""

# story: e13s03

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from vehicle_risk_agent.cli import predeploy_eval
from vehicle_risk_agent.evaluation.comparative import (
    ComparativeMode,
    ComparativeReport,
    build_offline_comparative_report,
    load_e12_evaluation_config,
)

NOTICE = (
    "NOTICE: offline predeploy gate — e12 release_verdict (requires live runs) is "
    "deliberately ignored; this is not the full e12 result."
)
FORCED = "FORCED FAILURE: --force-fail requested; deploy blocked deliberately"


@pytest.fixture(scope="module")
def real_report() -> ComparativeReport:
    return asyncio.run(build_offline_comparative_report(load_e12_evaluation_config()))


def _use_report(monkeypatch: pytest.MonkeyPatch, report: ComparativeReport) -> None:
    async def fake_builder(*_: Any, **__: Any) -> ComparativeReport:
        return report

    monkeypatch.setattr(predeploy_eval, "build_offline_comparative_report", fake_builder)


def test_real_frozen_config_passes(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = predeploy_eval.main([])

    out = capsys.readouterr()
    assert exit_code == 0, out.err
    assert NOTICE in out.out.splitlines()
    assert out.err == ""


def test_pinned_digest_matches_tracked_frozen_config(real_report: ComparativeReport) -> None:
    assert real_report.config_sha256 == predeploy_eval.EXPECTED_CONFIG_SHA256


def test_force_fail_exits_3_without_evaluating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def must_not_run(*_: Any, **__: Any) -> ComparativeReport:
        raise AssertionError("report builder must not run with --force-fail")

    monkeypatch.setattr(predeploy_eval, "build_offline_comparative_report", must_not_run)

    exit_code = predeploy_eval.main(["--force-fail"])

    out = capsys.readouterr()
    assert exit_code == 3
    assert FORCED in out.err


def _failing_baseline_metric(report: ComparativeReport) -> ComparativeReport:
    metrics = list(report.metrics)
    index = next(
        i for i, metric in enumerate(metrics) if metric.mode == ComparativeMode.OFFLINE_BASELINE
    )
    metrics[index] = metrics[index].model_copy(update={"deterministic_risk_passed": False})
    return report.model_copy(update={"metrics": tuple(metrics)})


@pytest.mark.parametrize(
    ("mutate", "check_name"),
    [
        (lambda r: r.model_copy(update={"config_sha256": "0" * 64}), "config_digest"),
        (_failing_baseline_metric, "offline_coverage"),
        (lambda r: r.model_copy(update={"offline_runs": 29}), "offline_coverage"),
        (
            lambda r: r.model_copy(update={"deterministic_risk_match": 0.9}),
            "deterministic_risk_match",
        ),
        (lambda r: r.model_copy(update={"unauthorized_dispatches": 1}), "unauthorized_dispatches"),
        (lambda r: r.model_copy(update={"automatic_approvals": 1}), "automatic_approvals"),
    ],
)
def test_failed_check_exits_1_and_names_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    real_report: ComparativeReport,
    mutate: Any,
    check_name: str,
) -> None:
    _use_report(monkeypatch, mutate(real_report))

    exit_code = predeploy_eval.main([])

    out = capsys.readouterr()
    assert exit_code == 1
    assert NOTICE in out.out.splitlines()
    assert check_name in out.err


def test_runner_exception_exits_2_with_class_name_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(*_: Any, **__: Any) -> ComparativeReport:
        raise RuntimeError("secret internal detail /etc/passwd")

    monkeypatch.setattr(predeploy_eval, "build_offline_comparative_report", broken)

    exit_code = predeploy_eval.main([])

    out = capsys.readouterr()
    assert exit_code == 2
    assert "RuntimeError" in out.err
    assert "secret internal detail" not in out.err + out.out
    assert "Traceback" not in out.err


def test_unknown_argument_exits_2() -> None:
    assert predeploy_eval.main(["--bogus"]) == 2


def test_output_file_writes_report_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, real_report: ComparativeReport
) -> None:
    _use_report(monkeypatch, real_report)
    target = tmp_path / "out" / "predeploy-report.json"

    assert predeploy_eval.main(["--output-file", str(target)]) == 0

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert ComparativeReport.model_validate(payload) == real_report

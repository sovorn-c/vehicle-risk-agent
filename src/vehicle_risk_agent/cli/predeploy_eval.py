"""Offline predeploy gate over the frozen e12 evaluation configuration.

Runs the 30 held-out scenarios through the deterministic workflow and judges
only offline evidence. The e12 ``release_verdict`` needs live runs and is
deliberately ignored; the gate says so on every evaluating run.

Exit codes: 0 pass, 1 check failed, 2 usage or runner error, 3 --force-fail.
"""

# story: e13s03

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeMode,
    ComparativeReport,
    E12EvaluationConfig,
    build_offline_comparative_report,
    load_e12_evaluation_config,
)

EXPECTED_CONFIG_SHA256 = "18bbd4634baebb1a0d99b7bed83577d2b2966a2e823019fbdfb2be89b6180efd"
NOTICE = (
    "NOTICE: offline predeploy gate — e12 release_verdict (requires live runs) is "
    "deliberately ignored; this is not the full e12 result."
)
FORCED_FAILURE = "FORCED FAILURE: --force-fail requested; deploy blocked deliberately"

EXIT_PASS = 0
EXIT_CHECK_FAILED = 1
EXIT_ERROR = 2
EXIT_FORCED = 3


def evaluate_checks(report: ComparativeReport, config: E12EvaluationConfig) -> dict[str, bool]:
    """Judge the offline report against the frozen thresholds only."""
    thresholds = config.thresholds
    baseline = [m for m in report.metrics if m.mode == ComparativeMode.OFFLINE_BASELINE]
    return {
        "config_digest": report.config_sha256 == EXPECTED_CONFIG_SHA256,
        "offline_coverage": (
            report.offline_runs == len(config.held_out_scenario_ids)
            and len(baseline) == len(config.held_out_scenario_ids)
            and all(m.quality_passed and m.deterministic_risk_passed for m in baseline)
        ),
        "deterministic_risk_match": (
            report.deterministic_risk_match >= thresholds.deterministic_risk_match
        ),
        "unauthorized_dispatches": (
            report.unauthorized_dispatches <= thresholds.unauthorized_dispatches
        ),
        "automatic_approvals": report.automatic_approvals <= thresholds.automatic_approvals,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predeploy-eval",
        description="Offline predeploy gate over the frozen e12 evaluation config.",
    )
    parser.add_argument(
        "--force-fail",
        action="store_true",
        help="Exit 3 without evaluating, to prove the deploy pipeline blocks.",
    )
    parser.add_argument("--output-file", type=Path, help="Write the report JSON to this path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate and return its exit code."""
    try:
        args = _parse_args(argv)
    except SystemExit:
        return EXIT_ERROR

    if args.force_fail:
        print(FORCED_FAILURE, file=sys.stderr)
        return EXIT_FORCED

    print(NOTICE)
    try:
        config = load_e12_evaluation_config()
        report = asyncio.run(build_offline_comparative_report(config))
        if args.output_file is not None:
            report.save_to_file(args.output_file)
    except Exception as exc:
        print(f"predeploy gate error: {type(exc).__name__}", file=sys.stderr)
        return EXIT_ERROR

    checks = evaluate_checks(report, config)
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"predeploy gate failed: {', '.join(failed)}", file=sys.stderr)
        return EXIT_CHECK_FAILED
    print("predeploy gate passed")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())

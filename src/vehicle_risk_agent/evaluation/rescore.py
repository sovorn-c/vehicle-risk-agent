"""No-network re-score command for updating a LIVE comparative report with human judgments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeReport,
    RescoreProvenance,
    _evaluation_input_digest,
    build_comparative_report,
    load_e12_evaluation_config,
    load_semantic_judgments,
)
from vehicle_risk_agent.evaluation.publication import (
    E12Publication,
    write_publication,
)


def rescore_report(
    report_path: str | Path,
    judgments_path: str | Path,
    output_file: str | Path | None = None,
    publication_file: str | Path | None = None,
) -> tuple[ComparativeReport, E12Publication | None]:
    """Re-score an existing LIVE report with operator semantic judgments without network calls."""
    target_report = Path(report_path)
    if not target_report.is_file():
        raise FileNotFoundError(f"Report file not found: {target_report}")

    report = ComparativeReport.load_from_file(target_report)

    if report.blocker_reason is not None:
        raise ValueError(
            f"Cannot re-score report with blocker_reason '{report.blocker_reason}': "
            "prerequisite-blocked reports cannot be re-scored."
        )

    if report.execution_mode != "LIVE":
        raise ValueError(
            f"Cannot re-score report with execution_mode '{report.execution_mode}': "
            "only LIVE reports can be re-scored."
        )

    if report.live_runs == 0 or not report.metrics:
        raise ValueError("Cannot re-score report with 0 live runs.")

    config = load_e12_evaluation_config().model_copy(
        update={
            "provider": report.provider,
            "model": report.model,
            "pricing_input_usd_per_million": report.pricing_input_usd_per_million,
            "pricing_output_usd_per_million": report.pricing_output_usd_per_million,
            "pricing_source_url": report.pricing_source_url,
            "pricing_checked_at": report.pricing_checked_at,
        }
    )

    recomputed_sha256 = _evaluation_input_digest(
        config,
        report.metrics,
        report.source_commit,
        report.config_sha256,
    )
    if recomputed_sha256 != report.evaluation_input_sha256:
        raise ValueError(
            "Recomputed evaluation_input_sha256 does not match report value: "
            f"expected {report.evaluation_input_sha256}, got {recomputed_sha256}"
        )

    target_judgments = Path(judgments_path)
    if not target_judgments.is_file():
        raise FileNotFoundError(f"Judgments file not found: {target_judgments}")

    judgments = load_semantic_judgments(target_judgments)
    for item in judgments:
        if item.evaluation_input_sha256 != report.evaluation_input_sha256:
            raise ValueError(
                f"Semantic judgment for scenario '{item.scenario_id}' has mismatched "
                f"evaluation_input_sha256: expected {report.evaluation_input_sha256}, "
                f"got {item.evaluation_input_sha256}"
            )

    rescored_from = RescoreProvenance(
        report_id=report.report_id,
        run_hash=report.run_hash,
        created_at=report.created_at,
    )

    rescored = build_comparative_report(
        config,
        report.metrics,
        semantic_judgments=judgments,
        execution_mode="LIVE",
        source_commit=report.source_commit,
        config_sha256=report.config_sha256,
        rescored_from=rescored_from,
    )

    out_path = Path(output_file or report_path)
    rescored.save_to_file(out_path)

    publication: E12Publication | None = None
    if publication_file:
        pub_path = Path(publication_file)
        publication = write_publication(out_path, pub_path)

    return rescored, publication


def build_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for report re-scoring."""
    parser = argparse.ArgumentParser(
        prog="vehicle_risk_agent.evaluation.rescore",
        description="Re-score an existing LIVE comparative report with human semantic judgments.",
    )
    parser.add_argument("--report", required=True, help="Path to existing comparative report JSON")
    parser.add_argument("--judgments", required=True, help="Path to semantic judgments YAML")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional output path for regenerated report JSON (defaults to overwriting --report)",
    )
    parser.add_argument(
        "--publication-file",
        default=None,
        help="Optional path to write matching publication JSON pointer",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    """Command-line interface entry point for report re-scoring."""
    parser = build_parser()
    parsed = parser.parse_args(args)
    try:
        rescored, _ = rescore_report(
            report_path=parsed.report,
            judgments_path=parsed.judgments,
            output_file=parsed.output_file,
            publication_file=parsed.publication_file,
        )
        sys.stdout.write(
            f"Report rescored successfully: verdict={rescored.release_verdict}, "
            f"semantic_status={rescored.semantic_status}\n"
        )
        return 0 if rescored.verdict_passed else 1
    except Exception as exc:
        sys.stderr.write(f"Error rescoring report: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

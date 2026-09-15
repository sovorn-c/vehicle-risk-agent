"""Artifact-bound publication for e12 comparative reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeReport,
    compute_report_run_hash,
)
from vehicle_risk_agent.evaluation.provenance import sha256_bytes, sha256_file


class E12Publication(BaseModel):
    """Small public pointer that is valid only for one exact report artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publication_version: str = "e12-publication-v1"
    report_id: str
    config_id: str
    config_hash: str
    report_hash: str
    report_bytes_sha256: str
    source_commit: str
    config_sha256: str
    judgments_sha256: str
    evaluation_input_sha256: str
    execution_mode: str
    release_verdict: str
    verdict_passed: bool
    synthetic_vehicle_evidence: bool


def publication_from_report(
    report: ComparativeReport,
    *,
    report_bytes_sha256: str | None = None,
) -> E12Publication:
    """Copy only non-sensitive identity and verdict fields from a report."""
    digest = report_bytes_sha256 or sha256_bytes(report.model_dump_json(indent=2).encode("utf-8"))
    return E12Publication(
        report_id=report.report_id,
        config_id=report.config_id,
        config_hash=report.config_hash,
        report_hash=report.run_hash,
        report_bytes_sha256=digest,
        source_commit=report.source_commit,
        config_sha256=report.config_sha256,
        judgments_sha256=report.judgments_sha256,
        evaluation_input_sha256=report.evaluation_input_sha256,
        execution_mode=report.execution_mode,
        release_verdict=report.release_verdict,
        verdict_passed=report.verdict_passed,
        synthetic_vehicle_evidence=report.synthetic_vehicle_evidence,
    )


def write_publication(
    report_path: str | Path,
    publication_path: str | Path,
) -> E12Publication:
    """Validate a report, bind its identity, and write a redacted pointer."""
    report = ComparativeReport.load_from_file(report_path)
    publication = publication_from_report(
        report,
        report_bytes_sha256=sha256_file(report_path),
    )
    target = Path(publication_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(publication.model_dump_json(indent=2), encoding="utf-8")
    return publication


def validate_publication(
    report_path: str | Path,
    publication_path: str | Path,
) -> E12Publication:
    """Reject a publication pointer that does not match its report artifact."""
    report = ComparativeReport.load_from_file(report_path)
    publication = E12Publication.model_validate_json(
        Path(publication_path).read_text(encoding="utf-8")
    )
    expected = publication_from_report(
        report,
        report_bytes_sha256=sha256_file(report_path),
    )
    if publication != expected:
        raise ValueError("e12 publication does not match the referenced report artifact")
    if report.run_hash != compute_report_run_hash(report):
        raise ValueError("e12 report hash does not match its contents")
    return publication


def main(args: list[str] | None = None) -> int:
    """Bind one report and print its public identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--publication-file", required=True)
    parsed = parser.parse_args(args)
    publication = write_publication(parsed.report, parsed.publication_file)
    print(publication.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

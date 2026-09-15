"""Contracts for artifact-bound e12 publication pointers."""

import asyncio
import json
from pathlib import Path

import pytest

from vehicle_risk_agent.evaluation.comparative import (
    ComparativeReport,
    build_offline_comparative_report,
)
from vehicle_risk_agent.evaluation.publication import (
    validate_publication,
    write_publication,
)


def test_publication_binds_config_and_report_hashes(tmp_path: Path) -> None:
    report_path = tmp_path / "e12-report.json"
    publication_path = tmp_path / "e12-publication.json"
    report = asyncio.run(build_offline_comparative_report())
    report.save_to_file(report_path)

    publication = write_publication(report_path, publication_path)

    assert publication.config_id == "e12-eval-v1"
    assert publication.config_hash == report.config_hash
    assert publication.report_hash == report.run_hash
    assert publication.release_verdict == "BLOCKED"
    assert len(publication.source_commit) == 40
    assert len(publication.config_sha256) == 64
    assert len(publication.judgments_sha256) == 64
    assert validate_publication(report_path, publication_path) == publication


def test_loading_rejects_tampered_report_with_stale_hash(tmp_path: Path) -> None:
    report_path = tmp_path / "e12-report.json"
    report = asyncio.run(build_offline_comparative_report())
    report.save_to_file(report_path)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["claim_support"] = 0.0
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="run_hash"):
        ComparativeReport.load_from_file(report_path)


def test_publication_rejects_pointer_for_a_different_report(tmp_path: Path) -> None:
    report_path = tmp_path / "e12-report.json"
    publication_path = tmp_path / "e12-publication.json"
    report = asyncio.run(build_offline_comparative_report())
    report.save_to_file(report_path)
    write_publication(report_path, publication_path)

    payload = json.loads(publication_path.read_text(encoding="utf-8"))
    payload["report_hash"] = "wrong"
    publication_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        validate_publication(report_path, publication_path)


def test_publication_rejects_wrong_source_commit(tmp_path: Path) -> None:
    report_path = tmp_path / "e12-report.json"
    publication_path = tmp_path / "e12-publication.json"
    report = asyncio.run(build_offline_comparative_report())
    report.save_to_file(report_path)
    write_publication(report_path, publication_path)

    payload = json.loads(publication_path.read_text(encoding="utf-8"))
    payload["source_commit"] = "0" * 40
    publication_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        validate_publication(report_path, publication_path)

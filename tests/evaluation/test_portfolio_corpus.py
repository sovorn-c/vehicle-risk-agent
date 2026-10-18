from pathlib import Path

import pytest
from vehicle_risk_agent.evaluation.portfolio_corpus import load_portfolio_corpus


def test_portfolio_corpus_is_pinned_attributable_and_bounded() -> None:
    bundle = load_portfolio_corpus(
        Path(__file__).parents[2] / "docs/benchmarks/e12/corpus/manifest.json"
    )

    assert bundle.manifest.id == "nz-vehicle-policy-portfolio-v1"
    assert len(bundle.sources) >= 4
    assert 15 <= len(bundle.passages) <= 50
    assert len({passage.id for passage in bundle.passages}) == len(bundle.passages)
    assert len(bundle.snapshot_ids) == len(bundle.sources)
    assert len(bundle.fingerprint) == 64
    for source, snapshot in zip(bundle.sources, bundle.snapshots, strict=True):
        assert source.canonical_origin.startswith("https://")
        assert source.issuing_authority
        assert source.reuse_terms
        assert snapshot.content_hash
        assert snapshot.passages
        assert all(passage.source_id == source.id for passage in snapshot.passages)
        assert all(passage.section_identifier for passage in snapshot.passages)


def test_portfolio_corpus_rejects_modified_source_content(tmp_path: Path) -> None:
    original = Path(__file__).parents[2] / "docs/benchmarks/e12/corpus"
    (tmp_path / "source.md").write_text("modified", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        (original / "manifest.json").read_text(encoding="utf-8").replace(
            "../sources/", "./"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash"):
        load_portfolio_corpus(tmp_path / "manifest.json")



def test_portfolio_corpus_loader_rejects_unknown_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema_version": 999}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_portfolio_corpus(manifest_path)

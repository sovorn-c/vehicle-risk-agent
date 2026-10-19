"""Opt-in end-to-end run of the frozen E12 retrieval comparison."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.evaluation.portfolio_corpus import PortfolioCorpusBundle
from vehicle_risk_agent.evaluation.portfolio_retrieval import (
    PortfolioMethodSummary,
    PortfolioQueryScore,
    PortfolioRetrievalDataset,
    score_portfolio_query,
    summarize_portfolio_results,
)
from vehicle_risk_agent.retrieval.adapters import (
    CrossEncoderRerankerAdapter,
    SentenceTransformersEmbeddingAdapter,
)
from vehicle_risk_agent.retrieval.postgres_index import PostgresPolicyIndex
from vehicle_risk_agent.retrieval.service import HybridRetrievalService


class SourceProvenance(TypedDict):
    commit_sha: str
    dirty: bool
    captured_at: str


def capture_source_provenance(repository: Path) -> SourceProvenance:
    """Capture source identity before measurement and before creating run artifacts.

    Git must be available; ignored local files are outside the tracked source tree.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    return {
        "commit_sha": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "captured_at": datetime.now(UTC).isoformat(),
    }


def _environment_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def _citation_passage_ids(result: Any) -> list[str]:
    return [citation.passage_id for citation in result.citations]


async def run_portfolio_retrieval_comparison(
    session: AsyncSession,
    corpus: PortfolioCorpusBundle,
    dataset: PortfolioRetrievalDataset,
) -> tuple[dict[str, list[PortfolioQueryScore]], dict[str, PortfolioMethodSummary]]:
    """Run keyword and configured hybrid retrieval on the identical evaluation inputs."""
    config = dataset.retrieval_config
    embedder = SentenceTransformersEmbeddingAdapter(
        model_name=config.embedding_model,
        revision=config.embedding_revision,
        dimensions=config.embedding_dimensions,
    )
    reranker = CrossEncoderRerankerAdapter(
        model_name=config.reranker_model,
        revision=config.reranker_revision,
    )
    index = PostgresPolicyIndex(
        session=session,
        embedder=embedder,
        snapshot_ids=corpus.snapshot_ids,
        config=config,
    )
    hybrid_service = HybridRetrievalService(index=index, reranker=reranker, config=config)
    known_passage_ids = {passage.id for passage in corpus.passages}
    results: dict[str, list[PortfolioQueryScore]] = {"keyword": [], "keyword_or": [], "hybrid": []}

    for query in (item for item in dataset.queries if item.split == "evaluation"):
        for method in ("keyword", "keyword_or"):
            try:
                keyword = await index.search_keyword(
                    query.query, top_k=dataset.top_k, any_term=method == "keyword_or"
                )
                keyword_ids = [candidate.passage_id for candidate in keyword]
                results[method].append(
                    score_portfolio_query(
                        query,
                        keyword_ids,
                        keyword_ids,
                        is_abstention=not keyword_ids,
                        known_passage_ids=known_passage_ids,
                        top_k=dataset.top_k,
                    )
                )
            except Exception as exc:
                results[method].append(
                    score_portfolio_query(
                        query,
                        [],
                        [],
                        is_abstention=None,
                        known_passage_ids=known_passage_ids,
                        technical_error=type(exc).__name__,
                        top_k=dataset.top_k,
                    )
                )

        try:
            hybrid = await hybrid_service.retrieve(query.query)
            hybrid_ids = _citation_passage_ids(hybrid)
            cited_ids = hybrid_ids
            results["hybrid"].append(
                score_portfolio_query(
                    query,
                    hybrid_ids,
                    cited_ids,
                    is_abstention=hybrid.is_abstention,
                    known_passage_ids=known_passage_ids,
                    top_k=dataset.top_k,
                )
            )
        except Exception as exc:
            results["hybrid"].append(
                score_portfolio_query(
                    query,
                    [],
                    [],
                    is_abstention=None,
                    known_passage_ids=known_passage_ids,
                    technical_error=type(exc).__name__,
                    top_k=dataset.top_k,
                )
            )

    summaries = {
        method: summarize_portfolio_results(method, dataset, results[method])
        for method in ("keyword", "keyword_or", "hybrid")
    }
    return results, summaries


def write_portfolio_retrieval_run(
    output_directory: Path,
    corpus: PortfolioCorpusBundle,
    dataset: PortfolioRetrievalDataset,
    results: dict[str, list[PortfolioQueryScore]],
    summaries: dict[str, PortfolioMethodSummary],
    *,
    source_provenance: SourceProvenance,
) -> tuple[Path, Path]:
    """Write an immutable per-run JSON bundle and flat per-query CSV."""
    output_directory.mkdir(parents=True, exist_ok=False)
    run_document = {
        "schema_version": 2,
        "run_id": output_directory.name,
        "created_at": datetime.now(UTC).isoformat(),
        "execution_status": (
            "complete"
            if all(summary.execution_status == "complete" for summary in summaries.values())
            else "partial"
        ),
        "corpus_fingerprint": corpus.fingerprint,
        "source_provenance": source_provenance,
        "corpus_size": {
            "source_count": len(corpus.sources),
            "passage_count": len(corpus.passages),
            "passage_text_characters": sum(len(p.text) for p in corpus.passages),
        },
        "method_definitions": {
            "keyword": "English plainto_tsquery (all terms), ts_rank_cd, no score threshold",
            "keyword_or": "English websearch_to_tsquery (quoted words joined by OR), "
            "ts_rank_cd, no score threshold",
            "hybrid": "Dense plus conjunctive keyword, RRF, cross-encoder reranking "
            "and configured abstention thresholds",
        },
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint,
        "top_k": dataset.top_k,
        "retrieval_config": dataset.retrieval_config.model_dump(mode="json"),
        "environment": _environment_fingerprint(),
        "methods": {
            method: {
                "summary": summary.model_dump(mode="json"),
                "queries": [score.model_dump(mode="json") for score in scores],
            }
            for method, (scores, summary) in (
                (method, (results[method], summary)) for method, summary in summaries.items()
            )
        },
        "targets": dataset.targets.model_dump(mode="json"),
        "precision_at_5_ceiling": dataset.precision_at_5_ceiling,
        "limitations": [
            "Curated 20-query portfolio benchmark; not a statistical significance study.",
            "Labels were author checked; independent legal review is not claimed.",
        ],
    }
    json_path = output_directory / "results.json"
    json_path.write_text(
        json.dumps(run_document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_path = output_directory / "per-query.csv"
    fields = [
        "method",
        "query_id",
        "answerable",
        "relevant_passage_ids",
        "retrieved_passage_ids",
        "citation_passage_ids",
        "recall_at_5",
        "precision_at_5",
        "reciprocal_rank_at_5",
        "is_abstention",
        "correct_abstention",
        "false_abstention",
        "citation_identity_valid",
        "technical_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method, scores in results.items():
            for score in scores:
                row = score.model_dump(mode="json")
                for field in (
                    "relevant_passage_ids",
                    "retrieved_passage_ids",
                    "citation_passage_ids",
                ):
                    row[field] = json.dumps(row[field], separators=(",", ":"))
                row["reciprocal_rank_at_5"] = row.pop("reciprocal_rank_at_5")
                writer.writerow(
                    {
                        "method": method,
                        **{field: row[field] for field in fields if field != "method"},
                    }
                )
    return json_path, csv_path


def create_run_directory(root: Path) -> Path:
    """Create a UTC timestamped destination without replacing prior evidence."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}-{suffix:02d}"
        suffix += 1
    return candidate


def benchmark_requested() -> bool:
    return os.getenv("RUN_E12_RETRIEVAL_BENCHMARK") == "1"

"""Frozen question set and arithmetic for the E12 portfolio retrieval comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from vehicle_risk_agent.evaluation.portfolio_corpus import PortfolioCorpusBundle
from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration


class PortfolioRetrievalQuery(BaseModel):
    """One immutable labeled query in the development or evaluation split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    split: Literal["development", "evaluation"]
    query: str = Field(min_length=1)
    relevant_passage_ids: tuple[str, ...] = ()
    expected_proposition: str | None = None
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def validate_answerability(self) -> PortfolioRetrievalQuery:
        if self.relevant_passage_ids:
            if not self.expected_proposition or self.unsupported_reason is not None:
                raise ValueError("Answerable queries need a proposition and no unsupported reason")
        elif self.expected_proposition is not None or not self.unsupported_reason:
            raise ValueError("Unsupported queries need a reason and no expected proposition")
        if len(self.relevant_passage_ids) != len(set(self.relevant_passage_ids)):
            raise ValueError("Relevant passage ids must be unique")
        return self


class PortfolioRetrievalTargets(BaseModel):
    """Predeclared target values for this version of the benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_5_min: float = Field(ge=0.0, le=1.0)
    mrr_at_5_min: float = Field(ge=0.0, le=1.0)
    unsupported_abstention_min: float = Field(ge=0.0, le=1.0)
    citation_identity_min: float = Field(ge=0.0, le=1.0)


class PortfolioRetrievalDataset(BaseModel):
    """Corpus-bound, versioned query labels and retrieval configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    label_author: str = Field(min_length=1)
    label_review: str = Field(min_length=1)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(gt=0)
    retrieval_config: RetrievalConfiguration
    targets: PortfolioRetrievalTargets
    queries: tuple[PortfolioRetrievalQuery, ...] = Field(min_length=1)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def precision_at_5_ceiling(self) -> float:
        answerable = [
            query
            for query in self.queries
            if query.split == "evaluation" and query.relevant_passage_ids
        ]
        if not answerable:
            return 0.0
        relevant_slots = sum(
            min(self.top_k, len(query.relevant_passage_ids)) for query in answerable
        )
        return round(relevant_slots / (len(answerable) * self.top_k), 4)


def load_portfolio_retrieval_dataset(
    dataset_path: Path,
    corpus: PortfolioCorpusBundle,
) -> PortfolioRetrievalDataset:
    """Load labels and reject split, corpus, target, or passage identity drift."""
    try:
        raw = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset = PortfolioRetrievalDataset.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid portfolio retrieval dataset: {exc}") from exc

    if dataset.schema_version != 1:
        raise ValueError(
            f"Unsupported portfolio retrieval schema version: {dataset.schema_version}"
        )
    if dataset.corpus_fingerprint != corpus.fingerprint:
        raise ValueError("Portfolio retrieval dataset is bound to a different corpus fingerprint")
    if dataset.retrieval_config != corpus.manifest.retrieval_config:
        raise ValueError("Portfolio retrieval settings differ from the pinned corpus settings")
    if dataset.top_k != 5:
        raise ValueError("Portfolio retrieval benchmark requires top_k=5")

    expected_targets = PortfolioRetrievalTargets(
        recall_at_5_min=0.90,
        mrr_at_5_min=0.80,
        unsupported_abstention_min=0.80,
        citation_identity_min=1.00,
    )
    if dataset.targets != expected_targets:
        raise ValueError("Portfolio retrieval target values are fixed for dataset v1")

    query_ids = [query.id for query in dataset.queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Portfolio retrieval query ids must be unique")

    passage_ids = {passage.id for passage in corpus.passages}
    missing_passage_ids = {
        passage_id
        for query in dataset.queries
        for passage_id in query.relevant_passage_ids
        if passage_id not in passage_ids
    }
    if missing_passage_ids:
        raise ValueError(
            "Portfolio retrieval labels reference passages outside the corpus: "
            + ", ".join(sorted(missing_passage_ids))
        )

    development = [query for query in dataset.queries if query.split == "development"]
    evaluation = [query for query in dataset.queries if query.split == "evaluation"]
    if len(development) != 6 or len(evaluation) != 20:
        raise ValueError(
            "Portfolio retrieval split must contain six development and 20 evaluation queries"
        )
    if sum(bool(query.relevant_passage_ids) for query in development) != 4:
        raise ValueError("Development split must contain four answerable queries")
    if sum(bool(query.relevant_passage_ids) for query in evaluation) != 15:
        raise ValueError("Evaluation split must contain 15 answerable queries")

    return dataset


class PortfolioQueryScore(BaseModel):
    """Per-query retrieval scores with technical failures kept distinct."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    split: Literal["development", "evaluation"]
    answerable: bool
    relevant_passage_ids: tuple[str, ...]
    retrieved_passage_ids: tuple[str, ...]
    citation_passage_ids: tuple[str, ...]
    is_abstention: bool | None
    technical_error: str | None
    recall_at_5: float | None
    precision_at_5: float | None
    reciprocal_rank_at_5: float | None
    correct_abstention: bool | None
    false_abstention: bool | None
    citation_identity_valid: bool | None


def score_portfolio_query(
    query: PortfolioRetrievalQuery,
    retrieved_passage_ids: Sequence[str],
    citation_passage_ids: Sequence[str],
    is_abstention: bool | None,
    known_passage_ids: set[str],
    technical_error: str | None = None,
    top_k: int = 5,
) -> PortfolioQueryScore:
    """Score one response using fixed top-k slots and explicit failure states."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    ranked_ids = tuple(retrieved_passage_ids[:top_k])
    cited_ids = tuple(citation_passage_ids[:top_k])
    answerable = bool(query.relevant_passage_ids)

    if technical_error is not None:
        return PortfolioQueryScore(
            query_id=query.id,
            split=query.split,
            answerable=answerable,
            relevant_passage_ids=query.relevant_passage_ids,
            retrieved_passage_ids=(),
            citation_passage_ids=(),
            is_abstention=None,
            technical_error=technical_error,
            recall_at_5=None,
            precision_at_5=None,
            reciprocal_rank_at_5=None,
            correct_abstention=None,
            false_abstention=None,
            citation_identity_valid=None,
        )

    if is_abstention is None:
        raise ValueError("Successful retrieval outcomes must specify abstention state")

    relevant_ids = set(query.relevant_passage_ids)
    retrieved_relevant = len(set(ranked_ids) & relevant_ids)
    if answerable:
        recall = retrieved_relevant / len(relevant_ids)
        precision = retrieved_relevant / top_k
        reciprocal_rank = 0.0
        for rank, passage_id in enumerate(ranked_ids, start=1):
            if passage_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        correct_abstention = None
        false_abstention = is_abstention
    else:
        recall = None
        precision = None
        reciprocal_rank = None
        correct_abstention = is_abstention and not ranked_ids
        false_abstention = None

    citation_identity_valid = (
        ranked_ids == cited_ids
        and len(ranked_ids) == len(set(ranked_ids))
        and all(passage_id in known_passage_ids for passage_id in ranked_ids)
    )

    return PortfolioQueryScore(
        query_id=query.id,
        split=query.split,
        answerable=answerable,
        relevant_passage_ids=query.relevant_passage_ids,
        retrieved_passage_ids=ranked_ids,
        citation_passage_ids=cited_ids,
        is_abstention=is_abstention,
        technical_error=None,
        recall_at_5=recall,
        precision_at_5=precision,
        reciprocal_rank_at_5=reciprocal_rank,
        correct_abstention=correct_abstention,
        false_abstention=false_abstention,
        citation_identity_valid=citation_identity_valid,
    )

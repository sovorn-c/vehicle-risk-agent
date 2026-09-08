"""Policy retrieval evaluation datasets, labels, and quality metrics."""

# story: e06s03

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyPassage
from vehicle_risk_agent.retrieval.service import HybridRetrievalService


class RetrievalQueryLabel(BaseModel):
    """Labelled search query with ground-truth relevant passages and citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    query: str
    relevant_passage_ids: tuple[str, ...] = ()
    required_citation_ids: tuple[str, ...] = ()
    is_no_answer: bool = False
    intent: str = ""


class RetrievalEvaluationDataset(BaseModel):
    """Versioned collection of labelled retrieval queries and corpus passages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    corpus_version: str
    queries: tuple[RetrievalQueryLabel, ...]
    passages: tuple[PolicyPassage, ...]


class RetrievalMetricsThresholds(BaseModel):
    """Quality thresholds for policy retrieval and citation grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_recall_at_5_min: float = 0.90
    context_precision_at_5_min: float = 0.70
    mrr_min: float = 0.80
    abstention_accuracy_min: float = 0.95
    citation_identity_accuracy: float = 1.00


class PerQueryRetrievalMetric(BaseModel):
    """Detailed retrieval metrics for a single evaluated query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    query: str
    is_no_answer: bool
    retrieved_passage_ids: tuple[str, ...]
    relevant_passage_ids: tuple[str, ...]
    required_citation_ids: tuple[str, ...]
    recall_at_5: float
    precision_at_5: float
    reciprocal_rank: float
    abstention_correct: bool
    citation_grounding_correct: bool


class RetrievalMetricsResult(BaseModel):
    """Aggregate evaluation report for retrieval and grounding quality gates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_recall_at_5: float
    context_precision_at_5: float
    mrr: float
    abstention_accuracy: float
    citation_identity_accuracy: float
    passed: bool
    query_metrics: tuple[PerQueryRetrievalMetric, ...]
    thresholds: RetrievalMetricsThresholds
    run_hash: str


class RagasEvaluationRecord(BaseModel):
    """Record schema compatible with Ragas single-turn dataset evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_input: str
    retrieved_contexts: list[str]
    response: str
    reference: str
    reference_contexts: list[str] = []


SEEDED_SOURCE_METADATA: dict[str, tuple[str, str]] = {
    "nz-legislation-fta-1986": (
        "Fair Trading Act 1986",
        "https://legislation.govt.nz/act/public/1986/0121/latest/DLM96439.html",
    ),
    "nz-legislation-cga-1993": (
        "Consumer Guarantees Act 1993",
        "https://legislation.govt.nz/act/public/1993/0091/latest/DLM311053.html",
    ),
    "nz-ppsr-act-1999": (
        "Personal Property Securities Act 1999",
        "https://legislation.govt.nz/act/public/1999/0126/latest/DLM45900.html",
    ),
    "nzta-virm-manual": (
        "NZTA Vehicle Inspection Requirements Manual",
        "https://vehicleinspection.nzta.govt.nz/virms/in-service-wof-and-coha",
    ),
}


def _make_passage(
    pid: str,
    snap_id: str,
    source_id: str,
    sec: str,
    heading: str,
    text: str,
    seq: int,
) -> PolicyPassage:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PolicyPassage(
        id=pid,
        snapshot_id=snap_id,
        source_id=source_id,
        section_identifier=sec,
        heading=heading,
        text=text,
        sequence=seq,
        char_offset_start=0,
        char_offset_end=len(text),
        content_hash=content_hash,
    )


def get_seeded_policy_passages() -> list[PolicyPassage]:
    """Return the authoritative set of policy passages for retrieval evaluation."""
    return [
        _make_passage(
            pid="snap-fta:p001",
            snap_id="snap-fta",
            source_id="nz-legislation-fta-1986",
            sec="Section 9",
            heading="Misleading and Deceptive Conduct in Trade",
            text=(
                "No person shall, in trade, engage in conduct that is misleading "
                "or deceptive regarding motor vehicle transactions."
            ),
            seq=1,
        ),
        _make_passage(
            pid="snap-fta:p002",
            snap_id="snap-fta",
            source_id="nz-legislation-fta-1986",
            sec="Section 13",
            heading="False Representations Concerning Vehicle History and Odometer",
            text=(
                "No person shall make false or misleading representations concerning "
                "vehicle history, repair status, or odometer readings."
            ),
            seq=2,
        ),
        _make_passage(
            pid="snap-cga:p001",
            snap_id="snap-cga",
            source_id="nz-legislation-cga-1993",
            sec="Section 6",
            heading="Guarantee as to Acceptable Quality",
            text=(
                "Where goods are supplied to a consumer, there is a statutory "
                "guarantee that goods are of acceptable quality, fit, and safe."
            ),
            seq=1,
        ),
        _make_passage(
            pid="snap-cga:p002",
            snap_id="snap-cga",
            source_id="nz-legislation-cga-1993",
            sec="Section 7",
            heading="Meaning of Acceptable Quality for Motor Vehicles",
            text=(
                "For motor vehicles, acceptable quality considers age, mileage, "
                "price, nature of defects, and pre-sale statements."
            ),
            seq=2,
        ),
        _make_passage(
            pid="snap-ppsr:p001",
            snap_id="snap-ppsr",
            source_id="nz-ppsr-act-1999",
            sec="Section 52",
            heading="Extinguishment of Security Interests",
            text=(
                "A buyer or lessee of motor vehicles takes free of unperfected "
                "security interest where extinguishment rules apply."
            ),
            seq=1,
        ),
        _make_passage(
            pid="snap-ppsr:p002",
            snap_id="snap-ppsr",
            source_id="nz-ppsr-act-1999",
            sec="Section 73",
            heading="Repossession Rights under Registered Security Interest",
            text=(
                "A registered security interest allows a secured creditor to "
                "repossess and sell the vehicle to satisfy debtor obligations."
            ),
            seq=2,
        ),
        _make_passage(
            pid="snap-virm:p001",
            snap_id="snap-virm",
            source_id="nzta-virm-manual",
            sec="Section 3-1",
            heading="Warrant of Fitness Inspection Standards",
            text=(
                "A valid Warrant of Fitness certification proves the vehicle "
                "satisfied roadworthiness inspection standards at examination time."
            ),
            seq=1,
        ),
        _make_passage(
            pid="snap-virm:p002",
            snap_id="snap-virm",
            source_id="nzta-virm-manual",
            sec="Section 2-4",
            heading="Statutory Write-Off Classifications and Deregistration",
            text=(
                "Vehicles classified as statutory write-offs suffer structural "
                "damage resulting in permanent deregistration without re-licensing."
            ),
            seq=2,
        ),
    ]


def get_seeded_retrieval_dataset() -> RetrievalEvaluationDataset:
    """Return the full labelled retrieval dataset containing answered and no-answer queries."""
    passages = get_seeded_policy_passages()
    queries = (
        RetrievalQueryLabel(
            query_id="q-01",
            query="misleading and deceptive conduct in motor vehicle trade",
            relevant_passage_ids=("snap-fta:p001",),
            required_citation_ids=("snap-fta:p001",),
            intent="Fair Trading Act section 9 misleading conduct",
        ),
        RetrievalQueryLabel(
            query_id="q-02",
            query="false representations concerning odometer readings and mileage tampering",
            relevant_passage_ids=("snap-fta:p002",),
            required_citation_ids=("snap-fta:p002",),
            intent="Fair Trading Act section 13 false odometer representations",
        ),
        RetrievalQueryLabel(
            query_id="q-03",
            query="statutory guarantee of acceptable quality for supplied consumer goods",
            relevant_passage_ids=("snap-cga:p001",),
            required_citation_ids=("snap-cga:p001",),
            intent="Consumer Guarantees Act section 6 acceptable quality guarantee",
        ),
        RetrievalQueryLabel(
            query_id="q-04",
            query="acceptable quality definition considering vehicle age mileage and defects",
            relevant_passage_ids=("snap-cga:p002",),
            required_citation_ids=("snap-cga:p002",),
            intent="Consumer Guarantees Act section 7 motor vehicle acceptable quality",
        ),
        RetrievalQueryLabel(
            query_id="q-05",
            query="buyer extinguishment of security interests on motor vehicle purchase",
            relevant_passage_ids=("snap-ppsr:p001",),
            required_citation_ids=("snap-ppsr:p001",),
            intent="PPSA section 52 extinguishment of security interest",
        ),
        RetrievalQueryLabel(
            query_id="q-06",
            query="repossession rights under registered security interest on motor vehicle finance",
            relevant_passage_ids=("snap-ppsr:p002",),
            required_citation_ids=("snap-ppsr:p002",),
            intent="PPSA section 73 repossession rights",
        ),
        RetrievalQueryLabel(
            query_id="q-07",
            query="warrant of fitness inspection standards and roadworthiness certification",
            relevant_passage_ids=("snap-virm:p001",),
            required_citation_ids=("snap-virm:p001",),
            intent="NZTA VIRM warrant of fitness inspection standards",
        ),
        RetrievalQueryLabel(
            query_id="q-08",
            query="statutory write-off catastrophic damage and permanent deregistration",
            relevant_passage_ids=("snap-virm:p002",),
            required_citation_ids=("snap-virm:p002",),
            intent="NZTA statutory write-off deregistration",
        ),
        RetrievalQueryLabel(
            query_id="q-09",
            query="secured creditor repossession of motor vehicle with outstanding finance",
            relevant_passage_ids=("snap-ppsr:p002",),
            required_citation_ids=("snap-ppsr:p002",),
            intent="PPSR finance repossession query",
        ),
        RetrievalQueryLabel(
            query_id="q-10",
            query="misleading representations about vehicle repair status and prior ownership",
            relevant_passage_ids=("snap-fta:p002",),
            required_citation_ids=("snap-fta:p002",),
            intent="FTA section 13 vehicle history misrepresentation",
        ),
        RetrievalQueryLabel(
            query_id="q-11",
            query="california air resources board carb emissions and executive order rules",
            relevant_passage_ids=(),
            required_citation_ids=(),
            is_no_answer=True,
            intent="Overseas regulatory standards with no NZ policy match",
        ),
        RetrievalQueryLabel(
            query_id="q-12",
            query="maritime commercial fishing vessel certificates and coastal maritime safety",
            relevant_passage_ids=(),
            required_citation_ids=(),
            is_no_answer=True,
            intent="Non-road vehicle domain with no policy match",
        ),
    )
    return RetrievalEvaluationDataset(
        dataset_id="nz-policy-eval-v1",
        corpus_version="corpus-2026.1",
        queries=queries,
        passages=tuple(passages),
    )


def compute_query_metrics(
    label: RetrievalQueryLabel,
    retrieved_passage_ids: list[str],
    retrieved_citation_ids: list[str],
    is_abstention: bool,
    k: int = 5,
) -> PerQueryRetrievalMetric:
    """Compute retrieval and grounding metrics for a single query."""
    top_k_ids = retrieved_passage_ids[:k]
    rel_set = set(label.relevant_passage_ids)
    req_cit_set = set(label.required_citation_ids)
    retrieved_cit_set = set(retrieved_citation_ids)

    if label.is_no_answer:
        abstention_correct = is_abstention and len(retrieved_passage_ids) == 0
        citation_correct = len(retrieved_cit_set) == 0
        return PerQueryRetrievalMetric(
            query_id=label.query_id,
            query=label.query,
            is_no_answer=True,
            retrieved_passage_ids=tuple(retrieved_passage_ids),
            relevant_passage_ids=label.relevant_passage_ids,
            required_citation_ids=label.required_citation_ids,
            recall_at_5=1.0 if abstention_correct else 0.0,
            precision_at_5=1.0 if abstention_correct else 0.0,
            reciprocal_rank=0.0,
            abstention_correct=abstention_correct,
            citation_grounding_correct=citation_correct,
        )

    abstention_correct = not is_abstention
    citation_correct = req_cit_set.issubset(retrieved_cit_set) if req_cit_set else True

    if not rel_set:
        recall = 1.0
        precision = 1.0
        rr = 1.0
    else:
        hits = [pid for pid in top_k_ids if pid in rel_set]
        recall = len(hits) / len(rel_set)

        rr = 0.0
        for rank, pid in enumerate(top_k_ids, start=1):
            if pid in rel_set:
                rr = 1.0 / rank
                break

        cumulative_prec = 0.0
        relevant_count = 0
        for rank, pid in enumerate(top_k_ids, start=1):
            if pid in rel_set:
                relevant_count += 1
                cumulative_prec += relevant_count / rank

        precision = (cumulative_prec / min(len(rel_set), k)) if relevant_count > 0 else 0.0

    return PerQueryRetrievalMetric(
        query_id=label.query_id,
        query=label.query,
        is_no_answer=False,
        retrieved_passage_ids=tuple(retrieved_passage_ids),
        relevant_passage_ids=label.relevant_passage_ids,
        required_citation_ids=label.required_citation_ids,
        recall_at_5=round(recall, 4),
        precision_at_5=round(precision, 4),
        reciprocal_rank=round(rr, 4),
        abstention_correct=abstention_correct,
        citation_grounding_correct=citation_correct,
    )


async def build_seeded_retrieval_service(
    dataset: RetrievalEvaluationDataset | None = None,
    config: RetrievalConfiguration | None = None,
    embedder: Any = None,
    reranker: Any = None,
) -> HybridRetrievalService:
    """Construct an in-memory HybridRetrievalService populated with seeded passages."""
    from vehicle_risk_agent.retrieval.adapters import FakeEmbeddingAdapter, FakeRerankerAdapter
    from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex

    ds = dataset or get_seeded_retrieval_dataset()
    cfg = config or RetrievalConfiguration(final_passage_cap=5, minimum_reranker_score=0.35)
    emb = embedder or FakeEmbeddingAdapter()
    rrk = reranker or FakeRerankerAdapter()
    index = InMemoryPolicyIndex(embedder=emb, config=cfg)
    await index.build_index(list(ds.passages))
    return HybridRetrievalService(
        index=index,
        reranker=rrk,
        config=cfg,
        source_metadata=SEEDED_SOURCE_METADATA,
    )


class RetrievalMetricsEvaluator:
    """Evaluates hybrid retrieval and grounding quality against pinned thresholds."""

    def __init__(
        self,
        service: Any,
        thresholds: RetrievalMetricsThresholds | None = None,
    ) -> None:
        self.service = service
        self.thresholds = thresholds or RetrievalMetricsThresholds()

    async def evaluate_dataset(
        self,
        dataset: RetrievalEvaluationDataset,
    ) -> RetrievalMetricsResult:
        """Evaluate each query in dataset and produce aggregate quality report."""
        query_metrics: list[PerQueryRetrievalMetric] = []

        for label in dataset.queries:
            result = await self.service.retrieve(label.query)
            retrieved_passage_ids = [c.passage_id for c in result.citations]
            retrieved_citation_ids = [c.passage_id for c in result.citations]
            metric = compute_query_metrics(
                label=label,
                retrieved_passage_ids=retrieved_passage_ids,
                retrieved_citation_ids=retrieved_citation_ids,
                is_abstention=result.is_abstention,
            )
            query_metrics.append(metric)

        answered = [m for m in query_metrics if not m.is_no_answer]
        no_answer = [m for m in query_metrics if m.is_no_answer]

        avg_recall = sum(m.recall_at_5 for m in answered) / len(answered) if answered else 1.0
        avg_precision = sum(m.precision_at_5 for m in answered) / len(answered) if answered else 1.0
        avg_mrr = sum(m.reciprocal_rank for m in answered) / len(answered) if answered else 1.0
        abstention_acc = (
            sum(1 for m in no_answer if m.abstention_correct) / len(no_answer) if no_answer else 1.0
        )
        citation_acc = (
            sum(1 for m in answered if m.citation_grounding_correct) / len(answered)
            if answered
            else 1.0
        )

        passed = (
            avg_recall >= self.thresholds.context_recall_at_5_min
            and avg_precision >= self.thresholds.context_precision_at_5_min
            and avg_mrr >= self.thresholds.mrr_min
            and abstention_acc >= self.thresholds.abstention_accuracy_min
            and citation_acc >= self.thresholds.citation_identity_accuracy
        )

        hash_payload = {
            "dataset_id": dataset.dataset_id,
            "corpus_version": dataset.corpus_version,
            "context_recall_at_5": round(avg_recall, 4),
            "context_precision_at_5": round(avg_precision, 4),
            "mrr": round(avg_mrr, 4),
            "abstention_accuracy": round(abstention_acc, 4),
            "citation_identity_accuracy": round(citation_acc, 4),
            "passed": passed,
            "queries": [
                {
                    "query_id": q.query_id,
                    "recall": q.recall_at_5,
                    "precision": q.precision_at_5,
                    "reciprocal_rank": q.reciprocal_rank,
                    "abstention_correct": q.abstention_correct,
                    "citation_grounding_correct": q.citation_grounding_correct,
                }
                for q in query_metrics
            ],
        }
        run_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return RetrievalMetricsResult(
            context_recall_at_5=round(avg_recall, 4),
            context_precision_at_5=round(avg_precision, 4),
            mrr=round(avg_mrr, 4),
            abstention_accuracy=round(abstention_acc, 4),
            citation_identity_accuracy=round(citation_acc, 4),
            passed=passed,
            query_metrics=tuple(query_metrics),
            thresholds=self.thresholds,
            run_hash=run_hash,
        )


def export_ragas_records(
    dataset: RetrievalEvaluationDataset,
    retrieval_results: dict[str, Any],
    responses: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Export Ragas-compatible records for offline quality evaluation."""
    records: list[dict[str, Any]] = []
    responses = responses or {}
    passage_by_id = {p.id: p for p in dataset.passages}

    for q in dataset.queries:
        res = retrieval_results.get(q.query_id)
        retrieved_contexts: list[str] = []
        if res and hasattr(res, "citations"):
            for c in res.citations:
                p = passage_by_id.get(c.passage_id)
                if p:
                    retrieved_contexts.append(f"{p.heading}: {p.text}")
                else:
                    retrieved_contexts.append(f"{c.heading}: {c.section_identifier}")

        if q.is_no_answer:
            resp = responses.get(
                q.query_id,
                "No authoritative New Zealand policy provisions apply to this inquiry.",
            )
            ref = "No relevant New Zealand policy provisions exist for this query."
            ref_contexts: list[str] = []
        else:
            rel_passages = [
                passage_by_id[pid] for pid in q.relevant_passage_ids if pid in passage_by_id
            ]
            ref_contexts = [f"{p.heading}: {p.text}" for p in rel_passages]
            ref = "\n".join(ref_contexts) if ref_contexts else "Relevant NZ statutory provision."
            resp = responses.get(
                q.query_id,
                f"Under New Zealand law, {rel_passages[0].text}"
                if rel_passages
                else "Policy grounded finding.",
            )

        record = RagasEvaluationRecord(
            user_input=q.query,
            retrieved_contexts=retrieved_contexts,
            response=resp,
            reference=ref,
            reference_contexts=ref_contexts,
        )
        records.append(record.model_dump())

    return records

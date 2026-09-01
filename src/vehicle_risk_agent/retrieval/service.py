"""Hybrid Policy Retrieval Service with dense search, keyword search, RRF, and reranking."""

from dataclasses import dataclass

from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyCitation
from vehicle_risk_agent.retrieval.adapters import RerankerAdapter
from vehicle_risk_agent.retrieval.fusion import reciprocal_rank_fusion
from vehicle_risk_agent.retrieval.index import InMemoryPolicyIndex, RankedCandidate


class PolicyRetrievalError(Exception):
    """Raised when a technical failure occurs during policy search, embedding, or reranking."""


@dataclass(frozen=True)
class RetrievalResult:
    """Complete traceable result of a policy retrieval execution."""

    query: str
    dense_candidates: list[RankedCandidate]
    keyword_candidates: list[RankedCandidate]
    fused_candidates: list[RankedCandidate]
    reranked_candidates: list[RankedCandidate]
    citations: list[PolicyCitation]
    is_abstention: bool


class HybridRetrievalService:
    """Coordinates multi-stage hybrid retrieval for Policy Knowledge."""

    def __init__(
        self,
        index: InMemoryPolicyIndex,
        reranker: RerankerAdapter,
        config: RetrievalConfiguration | None = None,
        source_metadata: dict[str, tuple[str, str]]
        | None = None,  # source_id -> (title, canonical_origin)
    ) -> None:
        self.index = index
        self.reranker = reranker
        self.config = config or RetrievalConfiguration()
        self.source_metadata = source_metadata or {}

    async def retrieve(self, query: str) -> RetrievalResult:
        """Execute hybrid search pipeline: dense + keyword -> RRF -> rerank -> citations."""
        if not query or not query.strip():
            return RetrievalResult(
                query=query,
                dense_candidates=[],
                keyword_candidates=[],
                fused_candidates=[],
                reranked_candidates=[],
                citations=[],
                is_abstention=True,
            )

        try:
            # 1. Parallel candidate generation
            dense_candidates = await self.index.search_dense(
                query=query,
                top_k=self.config.dense_candidates,
            )
            keyword_candidates = await self.index.search_keyword(
                query=query,
                top_k=self.config.keyword_candidates,
            )

            if not dense_candidates and not keyword_candidates:
                return RetrievalResult(
                    query=query,
                    dense_candidates=[],
                    keyword_candidates=[],
                    fused_candidates=[],
                    reranked_candidates=[],
                    citations=[],
                    is_abstention=True,
                )

            # 2. Reciprocal Rank Fusion
            fused_candidates = reciprocal_rank_fusion(
                dense_candidates=dense_candidates,
                keyword_candidates=keyword_candidates,
                rrf_k=self.config.rrf_k,
                cap=self.config.fused_candidate_cap,
            )

            # 3. Bound candidates for reranking
            candidates_to_rerank = fused_candidates[: self.config.rerank_candidate_cap]
            texts_to_score = [
                f"{c.passage.heading}\n{c.passage.text}" for c in candidates_to_rerank
            ]

            # 4. Cross-Encoder Reranking
            scores = await self.reranker.rerank(query=query, texts=texts_to_score)

            reranked_candidates: list[RankedCandidate] = []
            for rank_idx, (candidate, score) in enumerate(
                sorted(
                    zip(candidates_to_rerank, scores, strict=False),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ):
                reranked_candidates.append(
                    RankedCandidate(
                        passage_id=candidate.passage_id,
                        passage=candidate.passage,
                        score=score,
                        rank=rank_idx + 1,
                    )
                )

            # 5. Filter by threshold (0.35) and cap at final_passage_cap (5)
            filtered_candidates = [
                c for c in reranked_candidates if c.score >= self.config.minimum_reranker_score
            ]
            top_passages = filtered_candidates[: self.config.final_passage_cap]

            if not top_passages:
                return RetrievalResult(
                    query=query,
                    dense_candidates=dense_candidates,
                    keyword_candidates=keyword_candidates,
                    fused_candidates=fused_candidates,
                    reranked_candidates=reranked_candidates,
                    citations=[],
                    is_abstention=True,
                )

            # 6. Generate citations
            citations: list[PolicyCitation] = []
            for c in top_passages:
                p = c.passage
                meta = self.source_metadata.get(p.source_id)
                if meta is None:
                    raise PolicyRetrievalError(
                        f"Missing authoritative source metadata for source {p.source_id}"
                    )
                title, origin = meta
                citations.append(
                    PolicyCitation(
                        passage_id=p.id,
                        snapshot_id=p.snapshot_id,
                        source_id=p.source_id,
                        section_identifier=p.section_identifier,
                        heading=p.heading,
                        source_title=title,
                        canonical_origin=origin,
                    )
                )

            return RetrievalResult(
                query=query,
                dense_candidates=dense_candidates,
                keyword_candidates=keyword_candidates,
                fused_candidates=fused_candidates,
                reranked_candidates=reranked_candidates,
                citations=citations,
                is_abstention=False,
            )
        except Exception as err:
            if isinstance(err, PolicyRetrievalError):
                raise
            raise PolicyRetrievalError(f"Technical policy retrieval failure: {err}") from err

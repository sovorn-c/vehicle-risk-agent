"""In-memory and vector search policy index for dense and keyword retrieval."""

import math
import re
from dataclasses import dataclass
from typing import Protocol

from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyPassage
from vehicle_risk_agent.retrieval.adapters import EmbeddingAdapter


@dataclass(frozen=True)
class RankedCandidate:
    """Individual retrieval candidate with rank and score."""

    passage_id: str
    passage: PolicyPassage
    score: float
    rank: int


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class PolicyIndex(Protocol):
    """Common retrieval index contract for in-memory and PostgreSQL backends."""

    async def search_dense(self, query: str, top_k: int = 20) -> list[RankedCandidate]:
        """Return dense candidates ordered by descending relevance."""
        ...

    async def search_keyword(self, query: str, top_k: int = 20) -> list[RankedCandidate]:
        """Return full-text candidates ordered by descending relevance."""
        ...

    async def get_source_metadata(self, source_id: str) -> tuple[str, str] | None:
        """Resolve authoritative source title and origin for citation grounding."""
        ...


class InMemoryPolicyIndex:
    """In-memory index supporting dense vector search and keyword search."""

    def __init__(
        self,
        embedder: EmbeddingAdapter,
        config: RetrievalConfiguration | None = None,
    ) -> None:
        self.embedder = embedder
        self.config = config or RetrievalConfiguration()
        self.passages: dict[str, PolicyPassage] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.corpus_tokens: dict[str, list[str]] = {}

    async def build_index(self, passages: list[PolicyPassage]) -> None:
        """Index a collection of policy passages."""
        self.passages.clear()
        self.embeddings.clear()
        self.corpus_tokens.clear()

        if not passages:
            return

        texts_to_embed: list[str] = []
        passage_ids: list[str] = []

        for p in passages:
            self.passages[p.id] = p
            full_text = f"{p.heading}\n{p.text}"
            texts_to_embed.append(full_text)
            passage_ids.append(p.id)
            self.corpus_tokens[p.id] = re.findall(r"\w+", full_text.lower())

        vectors = await self.embedder.embed_texts(texts_to_embed)
        for pid, vec in zip(passage_ids, vectors, strict=False):
            self.embeddings[pid] = vec

    async def search_dense(self, query: str, top_k: int = 20) -> list[RankedCandidate]:
        """Perform dense vector search using cosine similarity."""
        if not self.passages:
            return []

        query_vec = await self.embedder.embed_query(query)
        scored: list[tuple[str, float]] = []

        for pid, doc_vec in self.embeddings.items():
            sim = _cosine_similarity(query_vec, doc_vec)
            scored.append((pid, sim))

        # Sort by score, then ID, so equal scores are load-order independent.
        scored.sort(key=lambda item: (-item[1], item[0]))

        limit = min(top_k, self.config.dense_candidates)
        top_candidates = scored[:limit]

        return [
            RankedCandidate(
                passage_id=pid,
                passage=self.passages[pid],
                score=round(score, 6),
                rank=idx + 1,
            )
            for idx, (pid, score) in enumerate(top_candidates)
        ]

    async def get_source_metadata(self, _source_id: str) -> tuple[str, str] | None:
        """In-memory indexes have no authoritative source registry."""
        return None

    async def search_keyword(self, query: str, top_k: int = 20) -> list[RankedCandidate]:
        """Perform keyword search matching term frequencies and BM25-like overlap."""
        if not self.passages:
            return []

        query_tokens = re.findall(r"\w+", query.lower())
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []

        for pid, doc_tokens in self.corpus_tokens.items():
            if not doc_tokens:
                continue
            doc_token_set = set(doc_tokens)
            match_count = sum(1 for qt in query_tokens if qt in doc_token_set)
            if match_count == 0:
                continue

            # Term overlap score + exact phrase boost
            overlap_score = match_count / len(query_tokens)
            doc_len_penalty = 1.0 / (1.0 + 0.001 * len(doc_tokens))
            score = overlap_score * doc_len_penalty
            scored.append((pid, score))

        scored.sort(key=lambda item: (-item[1], item[0]))

        limit = min(top_k, self.config.keyword_candidates)
        top_candidates = scored[:limit]

        return [
            RankedCandidate(
                passage_id=pid,
                passage=self.passages[pid],
                score=round(score, 6),
                rank=idx + 1,
            )
            for idx, (pid, score) in enumerate(top_candidates)
        ]

"""Embedding and Reranker adapter protocols and deterministic test implementations."""

import asyncio
import hashlib
import math
import re
from typing import Protocol


class EmbeddingAdapter(Protocol):
    """Protocol for asynchronous text embedding models."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a batch of text passages."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Compute embedding for a single search query."""
        ...


class RerankerAdapter(Protocol):
    """Protocol for asynchronous cross-encoder rerankers."""

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score each text against the query, returning scores bounded in [0.0, 1.0]."""
        ...


def _deterministic_token_vector(text: str, dimensions: int = 384) -> list[float]:
    """Generate a normalized deterministic pseudo-embedding based on token hashes."""
    tokens = re.findall(r"\w+", text.lower())
    vec = [0.0] * dimensions
    if not tokens:
        return vec

    for token in tokens:
        # Hash token into 3 distinct dimension indices for rich representation
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx1 = h % dimensions
        idx2 = (h >> 16) % dimensions
        idx3 = (h >> 32) % dimensions
        weight = 1.0 / math.sqrt(len(tokens))
        vec[idx1] += weight
        vec[idx2] += weight * 0.5
        vec[idx3] += weight * 0.25

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


class FakeEmbeddingAdapter:
    """Deterministic offline embedding adapter for unit tests and local grading."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Compute deterministic vectors for texts."""
        return [_deterministic_token_vector(t, self.dimensions) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Compute deterministic vector for query."""
        return _deterministic_token_vector(query, self.dimensions)


class FakeRerankerAdapter:
    """Deterministic offline cross-encoder adapter with sigmoid score bounding."""

    async def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Score texts against query using combined lexical and semantic overlap."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        scores: list[float] = []

        for text in texts:
            text_tokens = set(re.findall(r"\w+", text.lower()))
            if not query_tokens or not text_tokens:
                scores.append(0.1)
                continue

            intersection = query_tokens.intersection(text_tokens)
            # Jaccard + token ratio
            overlap_ratio = len(intersection) / len(query_tokens)

            # Map to sigmoid-like score in range [0.1, 0.95]
            # Higher overlap -> high score > 0.35 threshold
            raw_logit = (overlap_ratio * 6.0) - 2.0
            sigmoid_score = 1.0 / (1.0 + math.exp(-raw_logit))
            scores.append(round(sigmoid_score, 4))

        return scores

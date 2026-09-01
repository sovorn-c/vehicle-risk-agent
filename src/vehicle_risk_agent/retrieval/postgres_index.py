"""PostgreSQL pgvector dense search and tsvector full-text search index."""

import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from vehicle_risk_agent.persistence.models import PolicyPassageRecord, PolicySourceRecord
from vehicle_risk_agent.policy.corpus_models import RetrievalConfiguration
from vehicle_risk_agent.policy.models import PolicyPassage
from vehicle_risk_agent.retrieval.adapters import EmbeddingAdapter
from vehicle_risk_agent.retrieval.index import RankedCandidate


def _passage_record_to_domain(record: PolicyPassageRecord) -> PolicyPassage:
    """Map PolicyPassageRecord to PolicyPassage domain model."""
    return PolicyPassage(
        id=record.id,
        snapshot_id=record.snapshot_id,
        source_id=record.source_id,
        section_identifier=record.section_identifier,
        heading=record.heading,
        text=record.text,
        sequence=record.sequence,
        char_offset_start=record.char_offset_start,
        char_offset_end=record.char_offset_end,
        content_hash=record.content_hash,
    )


class PostgresPolicyIndex:
    """Retrieval index backed by PostgreSQL pgvector embeddings and tsvector keyword search."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbeddingAdapter,
        snapshot_ids: list[str] | tuple[str, ...],
        config: RetrievalConfiguration | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._snapshot_ids = list(snapshot_ids)
        self._config = config or RetrievalConfiguration()

    async def get_source_metadata(self, source_id: str) -> tuple[str, str] | None:
        """Resolve citation metadata from the authoritative source table."""
        stmt = select(PolicySourceRecord.title, PolicySourceRecord.canonical_origin).where(
            PolicySourceRecord.id == source_id
        )
        result = await self._session.execute(stmt)
        row = result.one_or_none()
        if row is None:
            return None
        return str(row.title), str(row.canonical_origin)

    async def search_dense(self, query: str, top_k: int = 20) -> list[RankedCandidate]:
        """Search passages by embedding cosine similarity using pgvector <=> operator."""
        if not self._snapshot_ids:
            return []

        query_vector = await self._embedder.embed_query(query)

        # Use cosine distance operator <=> from pgvector
        distance_expr = PolicyPassageRecord.embedding.cosine_distance(query_vector)

        stmt = (
            select(PolicyPassageRecord, distance_expr.label("distance"))
            .where(
                PolicyPassageRecord.snapshot_id.in_(self._snapshot_ids),
                PolicyPassageRecord.embedding.is_not(None),
            )
            .order_by(sa.asc(distance_expr), PolicyPassageRecord.id)
            .limit(min(top_k, self._config.dense_candidates))
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        candidates: list[RankedCandidate] = []
        for rank, (record, distance) in enumerate(rows, start=1):
            passage = _passage_record_to_domain(record)
            # Cosine similarity = 1.0 - cosine_distance
            sim_score = max(0.0, 1.0 - float(distance))
            candidates.append(
                RankedCandidate(
                    passage_id=passage.id,
                    passage=passage,
                    score=round(sim_score, 4),
                    rank=rank,
                )
            )

        return candidates

    async def search_keyword(self, query: str, top_k: int = 20) -> list[RankedCandidate]:
        """Search passages using PostgreSQL full-text search with plainto_tsquery."""
        if not self._snapshot_ids:
            return []

        clean_query = query.strip()
        if not clean_query:
            return []

        # Construct full-text tsvector expression combining heading and text
        doc_expr = func.to_tsvector(
            "english",
            PolicyPassageRecord.heading + " " + PolicyPassageRecord.text,
        )
        query_expr = func.plainto_tsquery("english", clean_query)
        rank_expr = func.ts_rank_cd(doc_expr, query_expr)

        stmt = (
            select(PolicyPassageRecord, rank_expr.label("rank_score"))
            .where(
                PolicyPassageRecord.snapshot_id.in_(self._snapshot_ids),
                doc_expr.bool_op("@@")(query_expr),
            )
            .order_by(sa.desc(rank_expr), PolicyPassageRecord.id)
            .limit(min(top_k, self._config.keyword_candidates))
        )

        result = await self._session.execute(stmt)
        rows = result.all()

        candidates: list[RankedCandidate] = []
        for rank, (record, score_val) in enumerate(rows, start=1):
            passage = _passage_record_to_domain(record)
            score = float(score_val) if score_val is not None else 0.0
            candidates.append(
                RankedCandidate(
                    passage_id=passage.id,
                    passage=passage,
                    score=round(score, 4),
                    rank=rank,
                )
            )

        return candidates

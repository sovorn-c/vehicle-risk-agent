"""Reciprocal Rank Fusion (RRF) algorithm for combining dense and keyword retrieval candidates."""

from vehicle_risk_agent.retrieval.index import RankedCandidate


def reciprocal_rank_fusion(
    dense_candidates: list[RankedCandidate],
    keyword_candidates: list[RankedCandidate],
    rrf_k: int = 60,
    cap: int = 40,
) -> list[RankedCandidate]:
    """Fuse two ranked candidate lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(doc) = sum(1 / (k + rank_m(doc))) for all rankers m.
    """
    scores: dict[str, float] = {}
    passages = {}

    for c in dense_candidates:
        passages[c.passage_id] = c.passage
        scores[c.passage_id] = scores.get(c.passage_id, 0.0) + (1.0 / (rrf_k + c.rank))

    for c in keyword_candidates:
        passages[c.passage_id] = c.passage
        scores[c.passage_id] = scores.get(c.passage_id, 0.0) + (1.0 / (rrf_k + c.rank))

    # Sort descending by score; break ties deterministically by passage_id
    sorted_items = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_items = sorted_items[:cap]

    return [
        RankedCandidate(
            passage_id=pid,
            passage=passages[pid],
            score=round(score, 6),
            rank=idx + 1,
        )
        for idx, (pid, score) in enumerate(top_items)
    ]

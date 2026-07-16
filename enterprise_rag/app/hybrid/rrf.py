"""
Reciprocal Rank Fusion (RRF) and score normalization utilities.
"""
from __future__ import annotations
from app.config import get_settings
from app.models.schemas import RetrievedChunk
from app.observability import get_logger

log = get_logger(__name__)
settings = get_settings()


def normalize_scores(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    scores = [c.score for c in chunks]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    for chunk in chunks:
        chunk.score = (chunk.score - lo) / span if span > 1e-9 else 1.0
    return chunks


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    k: int | None = None,
    top_k: int | None = None,
) -> tuple[list[RetrievedChunk], dict[str, float]]:
    rrf_k = k or settings.RRF_K
    limit = top_k or settings.HYBRID_TOP_K

    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in chunk_store:
                chunk_store[cid] = chunk

    ordered = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    result: list[RetrievedChunk] = []
    for cid, fused in ordered[:limit]:
        chunk = chunk_store[cid].model_copy()
        chunk.score = fused
        chunk.retrieval_method = "fused"
        result.append(chunk)

    log.debug("RRF fusion complete", extra={
        "input_lists": len(ranked_lists),
        "unique_chunks": len(chunk_store),
        "returned": len(result),
    })
    return result, dict(ordered[:limit])

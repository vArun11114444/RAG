"""
app/services/retrieval/fusion.py
Reciprocal Rank Fusion (RRF) of dense + BM25 results.
Includes metadata filtering and score normalisation to [0, 1].
"""
from __future__ import annotations
from typing import Any

from app.models.schemas import MetadataFilter, RetrievedChunk
from app.services.observability.logger import get_logger

_log = get_logger(__name__)


# ── Metadata filtering ────────────────────────────────────────────────────────

_OPS = {
    "eq":       lambda v, fv: v == fv,
    "gte":      lambda v, fv: v >= fv,
    "lte":      lambda v, fv: v <= fv,
    "in":       lambda v, fv: v in fv,
    "contains": lambda v, fv: str(fv).lower() in str(v).lower(),
}


def apply_filters(
    chunks: list[RetrievedChunk],
    filters: list[MetadataFilter],
) -> list[RetrievedChunk]:
    if not filters:
        return chunks

    def _passes(chunk: RetrievedChunk) -> bool:
        meta: dict[str, Any] = chunk.metadata
        for f in filters:
            value = meta.get(f.field)
            if value is None:
                return False
            op = _OPS.get(f.operator)
            if op is None or not op(value, f.value):
                return False
        return True

    filtered = [c for c in chunks if _passes(c)]
    _log.debug("metadata_filter", before=len(chunks), after=len(filtered))
    return filtered


# ── Score normalisation ───────────────────────────────────────────────────────

def _minmax_norm(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    rng = hi - lo
    if rng == 0:
        return [1.0] * len(scores)
    return [(s - lo) / rng for s in scores]


def normalize_scores(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    dense_normed = _minmax_norm([c.dense_score for c in chunks])
    bm25_normed  = _minmax_norm([c.bm25_score  for c in chunks])
    for i, c in enumerate(chunks):
        c.dense_score = round(dense_normed[i], 6)
        c.bm25_score  = round(bm25_normed[i],  6)
    return chunks


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results:  list[RetrievedChunk],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """
    RRF score = Σ 1 / (k + rank_i)
    Merges two ranked lists; de-duplicates by chunk_id.
    """
    scores: dict[str, float] = {}
    lookup: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        lookup[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        if chunk.chunk_id not in lookup:
            lookup[chunk.chunk_id] = chunk
        else:
            # preserve bm25_score on the existing record
            lookup[chunk.chunk_id].bm25_score = chunk.bm25_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result: list[RetrievedChunk] = []
    for cid, rrf_score in ranked:
        c = lookup[cid].model_copy()
        c.rrf_score = round(rrf_score, 6)
        result.append(c)

    _log.debug(
        "rrf_fusion",
        dense_n=len(dense_results),
        bm25_n=len(bm25_results),
        merged_n=len(result),
    )
    return result

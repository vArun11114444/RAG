"""
Score normalisation and metadata filtering utilities.

Normalisation: min-max scaling to [0, 1] per retrieval method so that
BM25 raw scores (unbounded) and cosine distances (0–2) are comparable
before or after RRF fusion.
"""
from __future__ import annotations

from typing import Any

from app.models.schemas import RetrievedChunk
from app.observability import get_logger

log = get_logger(__name__)


# ── Score normalisation ───────────────────────────────────────────────────────

def normalize_scores(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Min-max normalise the .score field in-place across *chunks*.
    Returns the same list for chaining.
    """
    if not chunks:
        return chunks

    scores = [c.score for c in chunks]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        # All scores identical — set to 1.0
        return [c.model_copy(update={"score": 1.0}) for c in chunks]

    normalized = [
        c.model_copy(update={"score": (c.score - min_s) / (max_s - min_s)})
        for c in chunks
    ]
    log.debug(
        "Scores normalised",
        extra={"min": round(min_s, 4), "max": round(max_s, 4), "n": len(chunks)},
    )
    return normalized


def normalize_scores_by_group(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Normalise scores independently per retrieval_method group.
    Useful before RRF so each signal is on the same [0,1] scale.
    """
    groups: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        groups.setdefault(c.retrieval_method, []).append(c)

    normalised: list[RetrievedChunk] = []
    for method, group in groups.items():
        normalised.extend(normalize_scores(group))

    return normalised


# ── Metadata filtering ────────────────────────────────────────────────────────

def apply_metadata_filters(
    chunks: list[RetrievedChunk],
    filters: dict[str, Any],
) -> list[RetrievedChunk]:
    """
    Post-filter chunks by metadata key-value equality.
    Supports nested dot-notation keys: e.g. "author.name" → chunk.metadata["author"]["name"].
    Supports list membership: filter value is a list → chunk value must be IN that list.

    Examples:
        {"doc_type": "policy"}
        {"year": [2022, 2023, 2024]}
        {"author.department": "legal"}
    """
    if not filters:
        return chunks

    def _get_nested(meta: dict[str, Any], key: str) -> Any:
        parts = key.split(".")
        val: Any = meta
        for part in parts:
            if not isinstance(val, dict):
                return None
            val = val.get(part)
        return val

    result: list[RetrievedChunk] = []
    for chunk in chunks:
        match = True
        for key, expected in filters.items():
            actual = _get_nested(chunk.metadata, key)
            if isinstance(expected, list):
                if actual not in expected:
                    match = False
                    break
            else:
                if actual != expected:
                    match = False
                    break
        if match:
            result.append(chunk)

    log.debug(
        "Metadata filter applied",
        extra={
            "filters": filters,
            "before": len(chunks),
            "after": len(result),
        },
    )
    return result

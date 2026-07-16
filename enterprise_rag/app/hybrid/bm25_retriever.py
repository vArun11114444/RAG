"""
BM25 retriever backed by rank_bm25.
The index is built lazily from ChromaDB chunks so no extra store is needed.
Thread-safe rebuild via asyncio.Lock.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import get_settings
from app.models.schemas import RetrievedChunk
from app.observability import get_logger, ERRORS

log = get_logger(__name__)
settings = get_settings()


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi.
    The corpus is loaded from the existing ChromaDB collection.
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build/rebuild the index from a list of chunk dicts.
        Each dict must have at least: {chunk_id, document_id, text, source, metadata}.
        Call this after ingestion or at startup.
        """
        async with self._lock:
            log.info("Building BM25 index", extra={"n_chunks": len(chunks)})
            self._chunks = chunks
            corpus = [_tokenize(c["text"]) for c in chunks]
            self._index = BM25Okapi(
                corpus,
                k1=settings.BM25_K1,
                b=settings.BM25_B,
            )
            log.info("BM25 index ready")

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Return up to *top_k* chunks ranked by BM25.
        Applies metadata_filters as a post-filter (exact key-value match).
        """
        if self._index is None:
            log.warning("BM25 index not built — returning empty results")
            return []

        k = top_k or settings.BM25_TOP_K
        tokens = _tokenize(query)

        # BM25 scoring is CPU-bound; offload so we don't block the event loop
        scores: list[float] = await asyncio.to_thread(
            self._index.get_scores, tokens
        )

        # Pair scores with chunks and sort
        ranked = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for score, chunk in ranked:
            if len(results) >= k:
                break
            # Apply metadata filters
            if metadata_filters:
                meta = chunk.get("metadata", {})
                if not all(meta.get(k2) == v for k2, v in metadata_filters.items()):
                    continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    text=chunk["text"],
                    score=float(score),
                    source=chunk.get("source", ""),
                    page=chunk.get("page"),
                    metadata=chunk.get("metadata", {}),
                    retrieval_method="bm25",
                )
            )

        log.debug(
            "BM25 retrieval complete",
            extra={"query": query, "returned": len(results)},
        )
        return results

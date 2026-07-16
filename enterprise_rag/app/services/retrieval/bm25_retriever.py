"""
app/services/retrieval/bm25_retriever.py
BM25 sparse retriever – wraps rank_bm25 with async-compatible interface.
Index is rebuilt incrementally when new documents arrive.
"""
from __future__ import annotations
import asyncio
import re
from typing import Optional

from rank_bm25 import BM25Okapi

from app.models.schemas import RetrievedChunk
from app.services.observability.logger import get_logger

_log = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    Maintains a BM25 index over document chunks.
    Thread-safe for reads; acquire _lock for writes.
    """

    def __init__(self) -> None:
        self._chunks: list[RetrievedChunk] = []
        self._index: Optional[BM25Okapi] = None
        self._lock = asyncio.Lock()

    # ── Index management ──────────────────────────────────────────────────────

    async def index_chunks(self, chunks: list[RetrievedChunk]) -> None:
        async with self._lock:
            self._chunks = chunks
            corpus = [_tokenize(c.content) for c in chunks]
            self._index = BM25Okapi(corpus)
            _log.info("bm25_index_built", num_chunks=len(chunks))

    async def add_chunks(self, new_chunks: list[RetrievedChunk]) -> None:
        async with self._lock:
            self._chunks.extend(new_chunks)
            corpus = [_tokenize(c.content) for c in self._chunks]
            self._index = BM25Okapi(corpus)
            _log.info("bm25_index_updated", total_chunks=len(self._chunks))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        if not self._index:
            _log.warning("bm25_index_empty")
            return []

        tokens = _tokenize(query)
        scores: list[float] = self._index.get_scores(tokens).tolist()

        ranked = sorted(
            zip(scores, self._chunks),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        results = []
        for score, chunk in ranked:
            c = chunk.model_copy()
            c.bm25_score = float(score)
            results.append(c)

        _log.debug("bm25_retrieved", query=query, top_k=len(results))
        return results

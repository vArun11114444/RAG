"""
app/services/retrieval/hybrid_retriever.py

MIGRATION: ChromaDB → Qdrant Cloud.
All business logic (BM25, RRF, query expansion, metadata filtering) unchanged.
Only the dense retrieval backend is swapped.
"""
from __future__ import annotations
import time
import asyncio

from app.core.config import get_settings
from app.models.schemas import (
    MetadataFilter,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.query_expander import QueryExpander
from app.services.retrieval.fusion import (
    apply_filters,
    normalize_scores,
    reciprocal_rank_fusion,
)
from app.services.observability.logger import get_logger
from app.services.observability.metrics import RETRIEVAL_LATENCY, RRF_TOP_SCORE
from app.services.observability.tracing import trace_span
from app.vector_store.qdrant_store import QdrantVectorStore

_settings = get_settings()
_log = get_logger(__name__)


class HybridRetriever:

    def __init__(self, bm25: BM25Retriever, expander: QueryExpander) -> None:
        self._bm25 = bm25
        self._expander = expander
        self._qdrant = QdrantVectorStore()

    # ── Dense retrieval via Qdrant ────────────────────────────────────────────

    async def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter],
    ) -> list[RetrievedChunk]:
        # Build ChromaDB-compatible where clause (QdrantVectorStore translates it)
        where: dict = {}
        for f in filters:
            if f.operator == "eq":
                where[f.field] = {"$eq": f.value}
            elif f.operator == "gte":
                where[f.field] = {"$gte": f.value}
            elif f.operator == "lte":
                where[f.field] = {"$lte": f.value}
            elif f.operator == "in":
                where[f.field] = {"$in": f.value}

        raw = await self._qdrant.query_by_text(
            query_texts=[query],
            n_results=top_k,
            where=where if where else None,
        )

        chunks: list[RetrievedChunk] = []
        ids   = raw.get("ids",   [[]])[0]
        docs  = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            chunks.append(RetrievedChunk(
                chunk_id=cid,
                content=doc,
                source=meta.get("source", ""),
                page=meta.get("page"),
                metadata=meta,
                dense_score=1.0 - float(dist),
            ))
        return chunks

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        request: RetrievalRequest,
        trace_id: str = "",
    ) -> RetrievalResult:
        t0 = time.perf_counter()

        async with trace_span("hybrid_retrieval", trace_id):
            queries = (
                await self._expander.expand(request.query)
                if request.expand_query
                else [request.query]
            )

            dense_tasks = [
                self._dense_retrieve(q, _settings.dense_top_k, request.filters)
                for q in queries
            ]
            bm25_tasks = [
                self._bm25.retrieve(q, _settings.bm25_top_k)
                for q in queries
            ]

            dense_lists, bm25_lists = await asyncio.gather(
                asyncio.gather(*dense_tasks),
                asyncio.gather(*bm25_tasks),
            )

            def _merge(lists):
                best: dict[str, RetrievedChunk] = {}
                for lst in lists:
                    for c in lst:
                        if c.chunk_id not in best or c.dense_score > best[c.chunk_id].dense_score:
                            best[c.chunk_id] = c
                return list(best.values())

            dense_flat = _merge(dense_lists)
            bm25_flat  = _merge(bm25_lists)
            bm25_flat  = apply_filters(bm25_flat, request.filters)

            normalize_scores(dense_flat)
            normalize_scores(bm25_flat)

            fused = reciprocal_rank_fusion(
                dense_flat, bm25_flat,
                k=_settings.rrf_k,
                top_k=request.top_k,
            )

            latency_ms = (time.perf_counter() - t0) * 1000
            RETRIEVAL_LATENCY.observe(latency_ms / 1000)
            if fused:
                RRF_TOP_SCORE.set(fused[0].rrf_score)

            _log.info(
                "hybrid_retrieval_complete",
                trace_id=trace_id,
                query=request.query,
                expanded=len(queries),
                chunks_returned=len(fused),
                latency_ms=round(latency_ms, 2),
            )

            return RetrievalResult(
                query=request.query,
                expanded_queries=queries[1:],
                chunks=fused,
                latency_ms=round(latency_ms, 2),
            )

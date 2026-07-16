"""
Hybrid Retrieval Orchestrator — Phase 1.

MIGRATION: ChromaDB → Qdrant Cloud.
Interface is identical to the previous ChromaDB version.
Accepts both:
  - retrieve(plan=QueryPlan)
  - retrieve(query=str, top_k=int, ...)
"""
from __future__ import annotations
import asyncio
from typing import Any
from app.config import get_settings
from app.models.schemas import (
    HybridRetrievalResult, QueryPlan, QueryType, RetrievedChunk, RetrievalStrategy
)
from app.observability import PHASE_LATENCY, RETRIEVAL_CHUNKS, Timer, get_logger, traced
from app.hybrid.bm25_retriever import BM25Retriever
from app.hybrid.metadata_filter import MetadataFilter, build_chroma_filter
from app.hybrid.query_expander import QueryExpander
from app.hybrid.rrf import normalize_scores, reciprocal_rank_fusion
from app.vector_store.qdrant_store import QdrantVectorStore

log = get_logger(__name__)
settings = get_settings()


class HybridRetriever:
    def __init__(self, qdrant: QdrantVectorStore | None = None,
                 bm25: BM25Retriever | None = None) -> None:
        self._qdrant = qdrant or QdrantVectorStore()
        self._bm25 = bm25 or BM25Retriever()
        self._expander = QueryExpander()

    async def initialise(self) -> None:
        """Load all chunks from Qdrant → build BM25 index."""
        try:
            await self._qdrant.ensure_collection()
            all_data = await self._qdrant.get_all()
            chunks = [
                {
                    "chunk_id":    id_,
                    "document_id": meta.get("document_id", id_),
                    "text":        doc,
                    "source":      meta.get("source", ""),
                    "page":        meta.get("page"),
                    "metadata":    meta,
                }
                for id_, doc, meta in zip(
                    all_data["ids"], all_data["documents"], all_data["metadatas"]
                )
            ]
            await self._bm25.build_index(chunks)
            log.info("HybridRetriever initialised", extra={"n_chunks": len(chunks)})
        except Exception as exc:
            log.warning("HybridRetriever init partial failure", extra={"error": str(exc)})

    @traced("hybrid_retrieval")
    async def retrieve(
        self,
        plan: QueryPlan | None = None,
        query: str | None = None,
        top_k: int | None = None,
        metadata_filters: dict[str, Any] | None = None,
        expanded_queries: list[str] | None = None,
        strategy: RetrievalStrategy | None = None,
        expand: bool = True,
    ) -> HybridRetrievalResult:
        """Unified retrieve — accepts QueryPlan or raw kwargs."""
        if plan is not None:
            _query   = (plan.expanded_queries or [""])[0]
            _top_k   = top_k or settings.HYBRID_TOP_K
            _filters = plan.metadata_filters or {}
            _expanded = list(plan.expanded_queries or [_query])
        else:
            _query   = query or ""
            _top_k   = top_k or settings.HYBRID_TOP_K
            _filters = metadata_filters or {}
            _expanded = list(expanded_queries or [_query])

        latency: dict[str, float] = {}

        # 1. Query expansion
        if expand:
            with Timer("query_expansion", latency):
                variants = await self._expander.expand(_query)
                seen = set(_expanded)
                for v in variants:
                    if v not in seen:
                        _expanded.append(v)
                        seen.add(v)

        # 2. Parallel BM25 + Qdrant vector retrieval
        with Timer("parallel_retrieval", latency):
            qdrant_filter = build_chroma_filter(_filters)   # same filter format
            bm25_results, vector_results = await asyncio.gather(
                self._bm25.retrieve(_query, top_k=settings.BM25_TOP_K,
                                    metadata_filters=_filters or None),
                self._vector_retrieve(queries=_expanded, top_k=settings.VECTOR_TOP_K,
                                      where=qdrant_filter),
            )

        # 3. Metadata post-filter
        with Timer("metadata_filter", latency):
            if _filters:
                bm25_results = MetadataFilter.apply(bm25_results, _filters)
                vector_results = MetadataFilter.apply(vector_results, _filters)

        # 4. Score normalization
        with Timer("score_normalization", latency):
            bm25_norm   = normalize_scores(list(bm25_results))
            vector_norm = normalize_scores(list(vector_results))

        # 5. RRF fusion
        with Timer("rrf_fusion", latency):
            fused, fused_scores = reciprocal_rank_fusion(
                bm25_norm, vector_norm, top_k=_top_k
            )

        RETRIEVAL_CHUNKS.labels(strategy="hybrid").observe(len(fused))
        for phase, ms in latency.items():
            PHASE_LATENCY.labels(phase=f"hybrid_{phase}").observe(ms / 1000)

        return HybridRetrievalResult(
            chunks=fused,
            bm25_scores={c.chunk_id: c.score for c in bm25_norm},
            vector_scores={c.chunk_id: c.score for c in vector_norm},
            fused_scores=fused_scores,
            query_variants=_expanded,
        )

    async def _vector_retrieve(self, queries: list[str], top_k: int,
                                where: dict | None) -> list[RetrievedChunk]:
        """Query Qdrant for each query variant, merge and deduplicate."""
        tasks = [
            self._qdrant.query_by_text(
                query_texts=[q],
                n_results=top_k,
                where=where,
            )
            for q in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen: set[str] = set()
        chunks: list[RetrievedChunk] = []
        for raw in results:
            if isinstance(raw, Exception):
                log.warning("Qdrant query failed", extra={"error": str(raw)})
                continue
            ids   = raw.get("ids", [[]])[0]
            docs  = raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0]
            dists = raw.get("distances", [[]])[0]
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                if cid in seen:
                    continue
                seen.add(cid)
                chunks.append(RetrievedChunk(
                    chunk_id=cid,
                    document_id=meta.get("document_id", cid),
                    text=doc,
                    score=1.0 - float(dist),
                    source=meta.get("source", ""),
                    page=meta.get("page"),
                    metadata=meta,
                    retrieval_method="vector",
                ))
        return chunks

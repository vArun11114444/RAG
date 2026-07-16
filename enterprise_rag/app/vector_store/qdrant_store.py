"""
app/vector_store/qdrant_store.py

Drop-in replacement for ChromaDB using Qdrant Cloud.

Preserves the full interface used by hybrid/retriever.py and
services/retrieval/hybrid_retriever.py:
  - collection creation / get-or-create
  - upsert (ingestion pipeline)
  - query (similarity search with metadata filtering)
  - get   (load all chunks for BM25 index build)

All ChromaDB-style metadata filters are transparently translated to
Qdrant Filter objects — business logic is untouched.

Connection:
  QDRANT_URL      = https://<cluster>.qdrant.io
  QDRANT_API_KEY  = your_api_key
  QDRANT_COLLECTION = documents
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from app.config import get_settings
from app.observability.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class QdrantVectorStore:
    """
    Async-safe Qdrant client wrapper.

    Qdrant's Python SDK is synchronous; we offload blocking calls to a
    thread pool via asyncio.to_thread so the FastAPI event loop stays free.
    """

    def __init__(self, url: str | None = None, api_key: str | None = None) -> None:
        self._url = url or settings.QDRANT_URL
        self._api_key = api_key or settings.QDRANT_API_KEY
        self._collection = settings.QDRANT_COLLECTION
        self._client = None
        self._vector_size: int = 384   # matches all-MiniLM-L6-v2 default

    def _get_client(self):
        """Lazy-initialise the synchronous Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=30,
            )
        return self._client

    # ── Collection management ─────────────────────────────────────────────────

    async def ensure_collection(
        self,
        collection_name: str | None = None,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> None:
        """Create collection if it does not already exist."""
        name = collection_name or self._collection
        self._vector_size = vector_size

        def _create():
            from qdrant_client.models import Distance, VectorParams
            client = self._get_client()
            existing = [c.name for c in client.get_collections().collections]
            if name not in existing:
                dist = {
                    "Cosine": Distance.COSINE,
                    "Dot":    Distance.DOT,
                    "Euclid": Distance.EUCLID,
                }.get(distance, Distance.COSINE)
                client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=dist),
                )
                log.info("qdrant_collection_created", extra={"name": name})
            else:
                log.info("qdrant_collection_exists", extra={"name": name})

        await asyncio.to_thread(_create)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        collection_name: str | None = None,
    ) -> None:
        """
        Upsert a batch of chunks into Qdrant.
        Called by the existing ingestion pipeline.
        """
        name = collection_name or self._collection

        def _upsert():
            from qdrant_client.models import PointStruct
            client = self._get_client()
            points = [
                PointStruct(
                    id=_str_to_uuid(cid),
                    vector=vec,
                    payload={
                        "chunk_id":   cid,
                        "document":   doc,
                        **meta,
                    },
                )
                for cid, vec, doc, meta in zip(ids, vectors, documents, metadatas)
            ]
            # Batch in chunks of 100 to stay within Qdrant free-tier limits
            for i in range(0, len(points), 100):
                client.upsert(collection_name=name, points=points[i:i + 100])

        await asyncio.to_thread(_upsert)
        log.info("qdrant_upserted", extra={"count": len(ids), "collection": name})

    # ── Similarity search ─────────────────────────────────────────────────────

    async def query(
        self,
        query_vector: list[float],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Search Qdrant and return a ChromaDB-compatible result dict:
            {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}

        This shape is intentional — the existing retriever code unpacks it
        the same way it did with ChromaDB, so no downstream changes needed.
        """
        name = collection_name or self._collection
        qdrant_filter = _build_qdrant_filter(where) if where else None

        def _search():
            from qdrant_client.models import Filter
            client = self._get_client()
            return client.search(
                collection_name=name,
                query_vector=query_vector,
                limit=n_results,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )

        try:
            hits = await asyncio.to_thread(_search)
        except Exception as exc:
            log.warning("qdrant_query_failed", extra={"error": str(exc)})
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        ids, docs, metas, dists = [], [], [], []
        for hit in hits:
            payload = hit.payload or {}
            ids.append(payload.get("chunk_id", str(hit.id)))
            docs.append(payload.get("document", ""))
            # Return all payload fields except internal ones as metadata
            meta = {k: v for k, v in payload.items() if k not in ("document", "chunk_id")}
            metas.append(meta)
            # Convert score (0-1 similarity) to distance (0=identical, 1=orthogonal)
            dists.append(1.0 - float(hit.score))

        return {
            "ids":       [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }

    async def query_by_text(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        collection_name: str | None = None,
        embedder=None,
    ) -> dict[str, Any]:
        """
        Text-based query — embeds the query first, then calls self.query().
        Used when the caller passes query_texts (ChromaDB-style interface).
        """
        if embedder is None:
            embedder = _get_default_embedder()

        # Use the first query text as primary (variants handled upstream via RRF)
        primary = query_texts[0] if query_texts else ""
        vector = await asyncio.to_thread(embedder.encode, primary)
        return await self.query(
            query_vector=vector.tolist(),
            n_results=n_results,
            where=where,
            collection_name=collection_name,
        )

    # ── Bulk get (for BM25 index build) ──────────────────────────────────────

    async def get_all(
        self,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Return all stored chunks in ChromaDB get() format:
            {"ids": [...], "documents": [...], "metadatas": [...]}

        Called by HybridRetriever.initialise() to build the BM25 index.
        Uses Qdrant scroll to page through all points.
        """
        name = collection_name or self._collection

        def _scroll_all():
            from qdrant_client.models import ScrollRequest
            client = self._get_client()
            all_ids, all_docs, all_metas = [], [], []
            offset = None
            while True:
                result, next_offset = client.scroll(
                    collection_name=name,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in result:
                    payload = point.payload or {}
                    all_ids.append(payload.get("chunk_id", str(point.id)))
                    all_docs.append(payload.get("document", ""))
                    meta = {k: v for k, v in payload.items()
                            if k not in ("document", "chunk_id")}
                    all_metas.append(meta)
                if next_offset is None:
                    break
                offset = next_offset
            return all_ids, all_docs, all_metas

        try:
            ids, docs, metas = await asyncio.to_thread(_scroll_all)
            log.info("qdrant_get_all", extra={"count": len(ids)})
            return {"ids": ids, "documents": docs, "metadatas": metas}
        except Exception as exc:
            log.warning("qdrant_get_all_failed", extra={"error": str(exc)})
            return {"ids": [], "documents": [], "metadatas": []}


# ── Filter translation ────────────────────────────────────────────────────────

def _build_qdrant_filter(where: dict[str, Any]):
    """
    Translate ChromaDB-style where clause → Qdrant Filter.

    Supported:
        {"field": "value"}                  → must match exactly
        {"field": {"$eq": v}}               → exact match
        {"field": {"$in": [v1, v2]}}        → any of
        {"field": {"$gte": v}}              → range
        {"field": {"$lte": v}}              → range
        {"$and": [{...}, {...}]}            → all must match
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchAny,
        MatchValue,
        Range,
    )

    def _condition(key: str, rule: Any) -> FieldCondition:
        if isinstance(rule, dict):
            if "$eq" in rule:
                return FieldCondition(key=key, match=MatchValue(value=rule["$eq"]))
            if "$in" in rule:
                return FieldCondition(key=key, match=MatchAny(any=rule["$in"]))
            rng: dict[str, Any] = {}
            if "$gte" in rule:
                rng["gte"] = rule["$gte"]
            if "$lte" in rule:
                rng["lte"] = rule["$lte"]
            if "$gt" in rule:
                rng["gt"] = rule["$gt"]
            if "$lt" in rule:
                rng["lt"] = rule["$lt"]
            if rng:
                return FieldCondition(key=key, range=Range(**rng))
        # Plain value → exact match
        return FieldCondition(key=key, match=MatchValue(value=rule))

    if "$and" in where:
        must = [_condition(list(c.keys())[0], list(c.values())[0]) for c in where["$and"]]
        return Filter(must=must)

    must = [_condition(k, v) for k, v in where.items()]
    return Filter(must=must) if must else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _str_to_uuid(s: str) -> str:
    """
    Qdrant requires UUID-format point IDs.
    Deterministically convert any string ID to a UUID v5.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


_embedder_instance = None

def _get_default_embedder():
    """Lazy-load sentence-transformers for text → vector conversion."""
    global _embedder_instance
    if _embedder_instance is None:
        from sentence_transformers import SentenceTransformer
        _embedder_instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder_instance

"""
app/services/knowledge_graph/graph_service.py
High-level facade: extract → store → traverse → expand context.
Called by the main pipeline when the planner enables KG retrieval.
"""
from __future__ import annotations
import time

from app.models.schemas import GraphContext, RetrievedChunk
from app.services.knowledge_graph.extractor import EntityRelationExtractor
from app.services.knowledge_graph.neo4j_store import Neo4jStore
from app.services.observability.logger import get_logger
from app.services.observability.metrics import GRAPH_LATENCY
from app.services.observability.tracing import trace_span

_log = get_logger(__name__)


class KnowledgeGraphService:

    def __init__(self, store: Neo4jStore, extractor: EntityRelationExtractor) -> None:
        self._store = store
        self._extractor = extractor

    async def build_context(
        self,
        chunks: list[RetrievedChunk],
        max_hops: int = 2,
        trace_id: str = "",
    ) -> GraphContext:
        t0 = time.perf_counter()

        async with trace_span("knowledge_graph", trace_id):
            # 1. Extract entities + relationships from retrieved chunks
            entities, rels = await self._extractor.extract(chunks)

            # 2. Upsert into Neo4j
            await self._store.upsert_entities(entities)
            await self._store.upsert_relationships(rels)

            # 3. Traverse neighbourhood for additional context
            entity_names = [e.name for e in entities]
            passages = await self._store.get_context_passages(
                entity_names, max_hops=max_hops
            )

            latency_ms = (time.perf_counter() - t0) * 1000
            GRAPH_LATENCY.observe(latency_ms / 1000)

            _log.info(
                "kg_context_built",
                trace_id=trace_id,
                entities=len(entities),
                relationships=len(rels),
                passages=len(passages),
                latency_ms=round(latency_ms, 2),
            )

            return GraphContext(
                entities=entities,
                relationships=rels,
                expanded_passages=passages,
                traversal_depth=max_hops,
            )

"""
Graph traversal and context expansion — Phase 2 entry point.

Given retrieved chunks, this module:
  1. Extracts entities from chunk text
  2. Finds matching entities in Neo4j
  3. Traverses the graph N hops
  4. Returns expanded chunk IDs for additional context
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.config import get_settings
from app.models.schemas import (
    Entity,
    GraphContext,
    Relationship,
    RetrievedChunk,
)
from app.observability import get_logger, ERRORS, GRAPH_ENTITIES
from app.observability.tracer import traced

from .extractor import EntityRelationExtractor
from .neo4j_client import Neo4jClient

log = get_logger(__name__)
settings = get_settings()


class GraphPipeline:
    """
    Orchestrates entity extraction → Neo4j upsert → traversal → context expansion.
    Safe to call even when Neo4j is unavailable (returns empty GraphContext).
    """

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        extractor: EntityRelationExtractor | None = None,
    ) -> None:
        self._neo4j = neo4j or Neo4jClient()
        self._extractor = extractor or EntityRelationExtractor()

    def is_available(self) -> bool:
        """Returns True if Neo4j is connected and graph features are active."""
        return self._neo4j.available

    @traced("graph_pipeline")
    async def run(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        max_hops: int | None = None,
        store_to_graph: bool = False,
    ) -> GraphContext:
        """
        Query-time graph enrichment. Alias for enrich() with store_to_graph=False.
        """
        return await self.enrich(chunks, max_hops=max_hops, store_to_graph=store_to_graph)

    @traced("graph_enrich")
    async def enrich(
        self,
        chunks: list[RetrievedChunk],
        max_hops: int | None = None,
        store_to_graph: bool = True,
    ) -> GraphContext:
        """
        Full graph enrichment for a set of retrieved chunks.

        Args:
            chunks:         Chunks retrieved by hybrid retrieval.
            max_hops:       How many hops to traverse from seed entities.
            store_to_graph: Whether to upsert extracted data into Neo4j.

        Returns:
            GraphContext with entities, relationships, and expanded chunk IDs.
        """
        hops = max_hops if max_hops is not None else settings.GRAPH_HOP_LIMIT

        if not self._neo4j.available:
            log.warning("Neo4j unavailable — skipping graph enrichment")
            return GraphContext()

        all_entities: list[Entity] = []
        all_rels: list[Relationship] = []

        try:
            all_entities, all_rels = await self._extractor.extract_batch(chunks)
        except Exception as exc:
            ERRORS.labels(phase="graph_extraction", error_type=type(exc).__name__).inc()
            log.error("Entity extraction failed", extra={"error": str(exc)})
            return GraphContext()

        if store_to_graph and all_entities:
            await asyncio.gather(
                self._neo4j.upsert_entities(all_entities),
                return_exceptions=True,
            )
        if store_to_graph and all_rels:
            await asyncio.gather(
                self._neo4j.upsert_relationships(all_rels),
                return_exceptions=True,
            )

        entity_texts = list({e.text for e in all_entities})
        seed_records = await self._neo4j.find_entities_by_text(entity_texts)
        seed_ids = [r["entity_id"] for r in seed_records]

        if not seed_ids:
            log.debug("No seed entities found in graph — no traversal")
            return GraphContext(entities=all_entities, relationships=all_rels)

        neighbor_records = await self._neo4j.traverse_neighbors(seed_ids, max_hops=hops)

        existing_ids = {c.chunk_id for c in chunks}
        expanded: list[str] = []
        max_depth_seen = 0
        for rec in neighbor_records:
            cid = rec.get("chunk_id")
            depth = rec.get("depth", 0)
            if cid and cid not in existing_ids:
                expanded.append(cid)
                existing_ids.add(cid)
                max_depth_seen = max(max_depth_seen, depth)

        log.info(
            "Graph enrichment complete",
            extra={
                "entities": len(all_entities),
                "relationships": len(all_rels),
                "seed_nodes": len(seed_ids),
                "expanded_chunks": len(expanded),
                "max_depth": max_depth_seen,
            },
        )

        return GraphContext(
            entities=all_entities,
            relationships=all_rels,
            expanded_chunk_ids=expanded,
            traversal_depth=max_depth_seen,
        )

    async def ingest_chunks(self, chunks: list[RetrievedChunk]) -> None:
        """Ingestion-time graph population — extracts and stores, no context returned."""
        await self.enrich(chunks, max_hops=0, store_to_graph=True)
        log.info("Graph ingestion complete", extra={"chunks": len(chunks)})

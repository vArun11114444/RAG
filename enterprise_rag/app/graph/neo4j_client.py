"""Neo4j async client for entity/relationship CRUD and graph traversal."""
from __future__ import annotations
from typing import Any
from app.config import get_settings
from app.models.schemas import Entity, GraphContext, Relationship
from app.observability import get_logger, GRAPH_TRAVERSAL_DEPTH

log = get_logger(__name__)
settings = get_settings()


class Neo4jClient:
    def __init__(self): self._driver = None

    async def connect(self):
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            await self._driver.verify_connectivity()
            await self._ensure_constraints()
            log.info("Neo4j connected")
        except Exception as exc:
            log.warning("Neo4j unavailable — graph disabled", extra={"error": str(exc)})
            self._driver = None

    async def close(self):
        if self._driver: await self._driver.close()

    @property
    def is_available(self): return self._driver is not None

    async def _ensure_constraints(self):
        async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
            await s.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
            await s.run("CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text)")

    async def upsert_entities(self, entities: list[Entity]):
        if not self.is_available or not entities: return
        async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
            await s.run("""
                UNWIND $entities AS e
                MERGE (n:Entity {entity_id: e.entity_id})
                SET n.label=e.label, n.text=e.text, n.confidence=e.confidence, n.source_chunk_id=e.source_chunk_id
            """, entities=[e.model_dump() for e in entities])

    async def upsert_relationships(self, relationships: list[Relationship]):
        if not self.is_available or not relationships: return
        async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
            await s.run("""
                UNWIND $rels AS r
                MATCH (src:Entity {entity_id: r.source_entity_id})
                MATCH (tgt:Entity {entity_id: r.target_entity_id})
                MERGE (src)-[rel:RELATES {rel_id: r.rel_id}]->(tgt)
                SET rel.relation_type=r.relation_type, rel.confidence=r.confidence, rel.source_chunk_id=r.source_chunk_id
            """, rels=[r.model_dump() for r in relationships])

    async def get_context_for_chunks(self, chunk_ids: list[str], hop_limit: int | None = None) -> GraphContext:
        if not self.is_available: return GraphContext()
        hops = hop_limit or settings.GRAPH_HOP_LIMIT
        cypher = """
            MATCH (seed:Entity) WHERE seed.source_chunk_id IN $chunk_ids
            OPTIONAL MATCH (seed)-[r*1..$hops]-(neighbor:Entity)
            RETURN collect(DISTINCT seed) + collect(DISTINCT neighbor) AS nodes,
                   collect(DISTINCT r) AS relationships
        """
        try:
            async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
                result = await s.run(cypher, chunk_ids=chunk_ids, hops=hops)
                records = await result.data()
        except Exception as exc:
            log.warning("Graph traversal failed", extra={"error": str(exc)})
            return GraphContext()

        entities, relationships = [], []
        expanded = set(chunk_ids)
        for record in records:
            for node in (record.get("nodes") or []):
                p = dict(node)
                expanded.add(p.get("source_chunk_id",""))
                entities.append(Entity(entity_id=p.get("entity_id",""), label=p.get("label","ENTITY"),
                    text=p.get("text",""), source_chunk_id=p.get("source_chunk_id",""), confidence=p.get("confidence",1.0)))
            for rel in (record.get("relationships") or []):
                if isinstance(rel, list):
                    for r in rel: _append_rel(r, relationships)
                else: _append_rel(rel, relationships)

        expanded.discard("")
        GRAPH_TRAVERSAL_DEPTH.observe(hops)
        return GraphContext(entities=entities, relationships=relationships,
                           expanded_chunk_ids=list(expanded), traversal_depth=hops)

    async def search_entities(self, text: str, limit: int = 10) -> list[Entity]:
        if not self.is_available: return []
        async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
            result = await s.run(
                "MATCH (e:Entity) WHERE toLower(e.text) CONTAINS toLower($text) RETURN e LIMIT $limit",
                text=text, limit=limit)
            rows = await result.data()
        return [Entity(entity_id=r["e"]["entity_id"], label=r["e"].get("label","ENTITY"),
            text=r["e"]["text"], source_chunk_id=r["e"].get("source_chunk_id",""),
            confidence=r["e"].get("confidence",1.0)) for r in rows]


def _append_rel(r: Any, target: list):
    try:
        p = dict(r)
        target.append(Relationship(rel_id=p.get("rel_id",""),
            source_entity_id=str(r.start_node.get("entity_id","")),
            target_entity_id=str(r.end_node.get("entity_id","")),
            relation_type=p.get("relation_type", r.type),
            source_chunk_id=p.get("source_chunk_id",""), confidence=p.get("confidence",1.0)))
    except Exception: pass


    @property
    def available(self) -> bool:
        """Alias for is_available — used by GraphPipeline."""
        return self._driver is not None

    async def find_entities_by_text(self, texts: list[str]) -> list[dict]:
        """Find entities by exact text match — used by GraphPipeline."""
        if not self.is_available or not texts:
            return []
        async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
            result = await s.run(
                "MATCH (e:Entity) WHERE e.text IN $texts RETURN e.entity_id AS entity_id, e.text AS text",
                texts=texts,
            )
            return await result.data()

    async def traverse_neighbors(self, entity_ids: list[str], max_hops: int = 2) -> list[dict]:
        """Traverse from seed entities and return neighboring chunk_ids."""
        if not self.is_available or not entity_ids:
            return []
        async with self._driver.session(database=settings.NEO4J_DATABASE) as s:
            result = await s.run(
                """
                MATCH (seed:Entity) WHERE seed.entity_id IN $ids
                OPTIONAL MATCH (seed)-[*1..$hops]-(neighbor:Entity)
                RETURN DISTINCT neighbor.source_chunk_id AS chunk_id, 
                       length(shortestPath((seed)-[*]-(neighbor))) AS depth
                """,
                ids=entity_ids, hops=max_hops,
            )
            return await result.data()

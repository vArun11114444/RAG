"""
app/services/knowledge_graph/neo4j_store.py
Neo4j async driver wrapper.  Handles upsert of entities/relationships
and multi-hop graph traversal for context expansion.
"""
from __future__ import annotations
import asyncio
from typing import Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.core.config import get_settings
from app.models.schemas import Entity, Relationship
from app.services.observability.logger import get_logger

_settings = get_settings()
_log = get_logger(__name__)


class Neo4jStore:

    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        self._driver = AsyncGraphDatabase.driver(
            _settings.neo4j_uri,
            auth=(_settings.neo4j_user, _settings.neo4j_password),
        )
        _log.info("neo4j_connected", uri=_settings.neo4j_uri)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            _log.info("neo4j_disconnected")

    # ── Upsert ────────────────────────────────────────────────────────────────

    async def upsert_entities(self, entities: list[Entity]) -> None:
        if not entities or not self._driver:
            return
        async with self._driver.session() as session:
            await session.execute_write(self._upsert_entities_tx, entities)

    @staticmethod
    async def _upsert_entities_tx(tx, entities: list[Entity]) -> None:
        cypher = """
        UNWIND $entities AS e
        MERGE (n {id: e.id})
        SET n.name = e.name, n.type = e.type
        SET n += e.properties
        WITH n, e
        CALL apoc.create.addLabels(n, [e.type]) YIELD node
        RETURN node
        """
        await tx.run(cypher, entities=[e.model_dump() for e in entities])

    async def upsert_relationships(self, rels: list[Relationship]) -> None:
        if not rels or not self._driver:
            return
        async with self._driver.session() as session:
            await session.execute_write(self._upsert_rels_tx, rels)

    @staticmethod
    async def _upsert_rels_tx(tx, rels: list[Relationship]) -> None:
        cypher = """
        UNWIND $rels AS r
        MATCH (a {id: r.source_id}), (b {id: r.target_id})
        MERGE (a)-[rel:RELATES_TO {type: r.relation_type}]->(b)
        SET rel += r.properties
        """
        await tx.run(cypher, rels=[r.model_dump() for r in rels])

    # ── Traversal ─────────────────────────────────────────────────────────────

    async def traverse(
        self,
        entity_names: list[str],
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict]:
        """
        Starting from named entities, traverse up to max_hops edges
        and return neighbouring node properties (for context expansion).
        """
        if not self._driver or not entity_names:
            return []

        cypher = """
        UNWIND $names AS name
        MATCH (start)
        WHERE toLower(start.name) CONTAINS toLower(name)
        CALL apoc.path.subgraphNodes(start, {
            maxLevel: $hops,
            limit: $limit
        }) YIELD node
        RETURN DISTINCT node.name AS name, node.type AS type,
               labels(node) AS labels, properties(node) AS props
        LIMIT $limit
        """
        try:
            async with self._driver.session() as session:
                result = await session.run(
                    cypher,
                    names=entity_names,
                    hops=max_hops,
                    limit=limit,
                )
                records = [r.data() async for r in result]
                _log.debug(
                    "graph_traversal",
                    entity_names=entity_names,
                    hops=max_hops,
                    records_found=len(records),
                )
                return records
        except Exception as exc:
            _log.warning("graph_traversal_failed", error=str(exc))
            return []

    async def get_context_passages(
        self,
        entity_names: list[str],
        max_hops: int = 2,
    ) -> list[str]:
        """Return human-readable context strings from graph neighbourhood."""
        nodes = await self.traverse(entity_names, max_hops=max_hops)
        passages: list[str] = []
        for n in nodes:
            name = n.get("name", "")
            ntype = n.get("type", "")
            props = {k: v for k, v in n.get("props", {}).items()
                     if k not in ("id", "name", "type") and v}
            if name:
                parts = [f"{name} ({ntype})"]
                if props:
                    parts.append(": " + "; ".join(f"{k}={v}" for k, v in list(props.items())[:3]))
                passages.append("".join(parts))
        return passages

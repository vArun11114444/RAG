"""
app/services/knowledge_graph/extractor.py
LLM-based entity and relationship extraction from retrieved text chunks.
Returns structured Entity + Relationship objects ready for Neo4j ingestion.
"""
from __future__ import annotations
import json
import uuid

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.schemas import Entity, Relationship, RetrievedChunk
from app.services.observability.logger import get_logger

_settings = get_settings()
_log = get_logger(__name__)

_SYSTEM = """You are an information extraction assistant.
Extract named entities and relationships from the provided text.

Return ONLY valid JSON with this structure (no markdown, no explanation):
{
  "entities": [
    {"name": "string", "type": "PERSON|ORG|LOCATION|CONCEPT|PRODUCT|DATE|REGULATION", "properties": {}}
  ],
  "relationships": [
    {"source": "entity name", "target": "entity name", "relation": "VERB_PHRASE", "properties": {}}
  ]
}
Keep entity names concise and canonical. Merge coreferences to the same entity name."""


class EntityRelationExtractor:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=_settings.openai_api_key, base_url=_settings.openai_base_url)

    async def extract(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[Entity], list[Relationship]]:
        combined_text = "\n\n".join(
            f"[Source: {c.source}]\n{c.content}" for c in chunks
        )
        if not combined_text.strip():
            return [], []

        try:
            resp = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": combined_text[:6000]},
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:
            _log.warning("extraction_failed", error=str(exc))
            return [], []

        # Map name → id so relationships can reference by id
        name_to_id: dict[str, str] = {}
        entities: list[Entity] = []
        for e in raw.get("entities", []):
            eid = f"ent_{uuid.uuid4().hex[:8]}"
            name_to_id[e.get("name", "")] = eid
            entities.append(Entity(
                id=eid,
                name=e.get("name", ""),
                type=e.get("type", "CONCEPT"),
                properties=e.get("properties", {}),
            ))

        relationships: list[Relationship] = []
        for r in raw.get("relationships", []):
            src = r.get("source", "")
            tgt = r.get("target", "")
            if src in name_to_id and tgt in name_to_id:
                relationships.append(Relationship(
                    source_id=name_to_id[src],
                    target_id=name_to_id[tgt],
                    relation_type=r.get("relation", "RELATED_TO"),
                    properties=r.get("properties", {}),
                ))

        _log.info(
            "extraction_complete",
            entities=len(entities),
            relationships=len(relationships),
        )
        return entities, relationships

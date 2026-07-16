"""
Entity and Relationship Extractor — Phase 2.
Uses spaCy for NER + LLM for relation triplet extraction.
"""
from __future__ import annotations
import asyncio, hashlib, json, re
from app.config import get_settings
from app.models.schemas import Entity, Relationship, RetrievedChunk
from app.observability import get_logger, GRAPH_ENTITIES

log = get_logger(__name__)
settings = get_settings()
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            log.warning("spaCy not available — using LLM-only extraction")
            _NLP = False
    return _NLP

_RELATION_PROMPT = """Extract relationship triplets from the text below.
Return ONLY a JSON array: [{{"subject":"...","predicate":"...","object":"..."}}]
Text: {text}"""

def _make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


class EntityRelationExtractor:
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self._llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL) if settings.OPENAI_API_KEY else None
        except ImportError:
            self._llm = None

    async def extract_from_chunks(self, chunks: list[RetrievedChunk]):
        results = await asyncio.gather(
            *[self._extract_single(c) for c in chunks], return_exceptions=True
        )
        all_entities, all_relations = [], []
        for r in results:
            if isinstance(r, Exception):
                continue
            ents, rels = r
            all_entities.extend(ents)
            all_relations.extend(rels)
        seen, unique = set(), []
        for e in all_entities:
            key = f"{e.text.lower()}|{e.label}"
            if key not in seen:
                seen.add(key); unique.append(e)
        GRAPH_ENTITIES.observe(len(unique))
        return unique, all_relations

    async def _extract_single(self, chunk):
        entities = await asyncio.to_thread(self._spacy_entities, chunk)
        relations = await self._llm_relations(chunk, entities)
        return entities, relations

    def _spacy_entities(self, chunk):
        nlp = _get_nlp()
        if not nlp: return []
        doc = nlp(chunk.text[:2000])
        return [Entity(
            entity_id=_make_id(f"{e.text}|{e.label_}|{chunk.chunk_id}"),
            label=e.label_, text=e.text, source_chunk_id=chunk.chunk_id, confidence=1.0,
        ) for e in doc.ents]

    async def _llm_relations(self, chunk, entities):
        if not self._llm or not entities: return []
        entity_map = {e.text.lower(): e.entity_id for e in entities}
        try:
            resp = await self._llm.chat.completions.create(
                model=settings.OPENAI_MODEL, temperature=0.0,
                messages=[{"role":"user","content":_RELATION_PROMPT.format(text=chunk.text[:1500])}],
                max_tokens=512,
            )
            raw = re.sub(r"```json|```","",resp.choices[0].message.content or "[]").strip()
            triplets = json.loads(raw)
        except Exception as exc:
            log.debug("LLM relation extraction failed", extra={"error": str(exc)})
            return []
        rels = []
        for t in triplets:
            subj, pred, obj = t.get("subject",""), t.get("predicate",""), t.get("object","")
            src_id = entity_map.get(subj.lower())
            tgt_id = entity_map.get(obj.lower())
            if src_id and tgt_id:
                rels.append(Relationship(
                    rel_id=_make_id(f"{src_id}|{pred}|{tgt_id}"),
                    source_entity_id=src_id, target_entity_id=tgt_id,
                    relation_type=pred, source_chunk_id=chunk.chunk_id, confidence=0.85,
                ))
        return rels

    async def extract_batch(self, chunks) -> tuple:
        """Alias for extract_from_chunks — used by GraphPipeline."""
        return await self.extract_from_chunks(chunks)

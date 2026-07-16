"""Query Planner — Phase 4. Classifies queries and produces a QueryPlan."""
from __future__ import annotations
import re
from typing import Any
from app.config import get_settings
from app.models.schemas import QueryPlan, QueryType, RetrievalStrategy
from app.observability import get_logger, traced

log = get_logger(__name__)
settings = get_settings()

_COMPLIANCE  = re.compile(r"\b(regulation|comply|compliance|gdpr|hipaa|sox|pci|audit|policy|legal|mandatory|obligation|breach|violation)\b", re.I)
_MULTI_HOP   = re.compile(r"\b(how does .+ affect|relationship between|compare .+ and|connection|chain of|sequence|trace|because of|leading to|result in)\b", re.I)
_KG          = re.compile(r"\b(who is connected|works with|org.?chart|related entities|what companies|which people|network of|graph of|collaborat|partner|subsidiary|acquired)\b", re.I)
_RESEARCH    = re.compile(r"\b(summarize|overview|landscape|state of the art|review|survey|comprehensive|all aspects|research|literature|history of|evolution)\b", re.I)

def _rule_classify(query: str) -> QueryType | None:
    q = query.lower()
    if _COMPLIANCE.search(q):  return QueryType.COMPLIANCE
    if _RESEARCH.search(q):    return QueryType.RESEARCH
    if _KG.search(q):          return QueryType.KNOWLEDGE_GRAPH
    if _MULTI_HOP.search(q):   return QueryType.MULTI_HOP
    if len(query.split()) <= 8 and "?" in query: return QueryType.SIMPLE
    return None

_CLASSIFY_PROMPT = """Classify this query into EXACTLY ONE of: simple, multi_hop, knowledge_graph, compliance, research.
Query: {query}
Reply with ONLY the type label, lowercase."""

_STRATEGY_MAP = {
    QueryType.SIMPLE:          (RetrievalStrategy.HYBRID,       False, False, "standard", 1),
    QueryType.MULTI_HOP:       (RetrievalStrategy.HYBRID,       True,  True,  "standard", 2),
    QueryType.KNOWLEDGE_GRAPH: (RetrievalStrategy.HYBRID_GRAPH, True,  True,  "standard", 3),
    QueryType.COMPLIANCE:      (RetrievalStrategy.HYBRID,       False, True,  "strict",   1),
    QueryType.RESEARCH:        (RetrievalStrategy.HYBRID_GRAPH, True,  True,  "standard", 2),
}


class QueryPlanner:
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self._llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL) if settings.OPENAI_API_KEY else None
        except ImportError:
            self._llm = None

    @traced("query_planner")
    async def plan(self, query: str, metadata_filters: dict[str, Any] | None = None,
                   force_type: QueryType | None = None) -> QueryPlan:
        qtype = force_type or _rule_classify(query) or await self._llm_classify(query)
        log.info("Query classified", extra={"type": qtype, "query": query[:80]})
        strategy, use_graph, verify, level, hops = _STRATEGY_MAP[qtype]
        return QueryPlan(
            query_type=qtype, retrieval_strategy=strategy,
            use_graph=use_graph, require_verification=verify,
            verification_level=level, max_hops=hops,
            metadata_filters=metadata_filters or {},
            expanded_queries=[query],
        )

    async def _llm_classify(self, query: str) -> QueryType:
        if not self._llm: return QueryType.SIMPLE
        try:
            resp = await self._llm.chat.completions.create(
                model=settings.OPENAI_MODEL, temperature=0.0,
                messages=[{"role":"user","content":_CLASSIFY_PROMPT.format(query=query)}],
                max_tokens=10,
            )
            return QueryType((resp.choices[0].message.content or "simple").strip().lower())
        except Exception as exc:
            log.debug("LLM classify failed", extra={"error": str(exc)})
            return QueryType.SIMPLE

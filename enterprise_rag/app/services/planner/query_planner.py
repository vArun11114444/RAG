"""
app/services/planner/query_planner.py
Phase 4: classifies the query and decides retrieval strategy,
graph usage, and verification depth.
"""
from __future__ import annotations
import json
import re

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.schemas import PlannerDecision, QueryType
from app.services.observability.logger import get_logger
from app.services.observability.tracing import trace_span

_settings = get_settings()
_log = get_logger(__name__)

_SYSTEM = """You are a query classification engine for an enterprise document retrieval system.

Classify the user query into exactly ONE of:
- simple_retrieval   : single-step factual lookup (e.g. "what is X?", "define Y")
- multi_hop          : requires chaining multiple retrieved facts (e.g. "how does A affect B through C?")
- knowledge_graph    : entity-centric, relationship-heavy (e.g. "who reports to X?", "list subsidiaries of Y")
- compliance         : regulatory, policy, legal, or audit questions (e.g. "does policy X permit Y?")
- research           : broad synthesis across many sources (e.g. "summarise all evidence for X")

Also decide:
- use_hybrid_retrieval : always true unless the query is purely a graph lookup
- use_knowledge_graph  : true for knowledge_graph, multi_hop, compliance, research
- verification_depth   : "minimal" (simple) | "standard" (multi_hop/compliance) | "deep" (research/compliance)
- max_hops             : 1 for simple/multi_hop, 2 for knowledge_graph, 3 for research
- reasoning            : one sentence explaining the classification

Return ONLY valid JSON:
{
  "query_type": "...",
  "use_hybrid_retrieval": true,
  "use_knowledge_graph": false,
  "verification_depth": "standard",
  "max_hops": 1,
  "reasoning": "..."
}"""

# Fast heuristic patterns to avoid an LLM call for obvious cases
_SIMPLE_PATTERNS = [
    r"^(what is|define|who is|what does|when was|where is)\s",
]
_KG_PATTERNS = [
    r"(who reports to|subsidiar|organisation chart|hierarchy|relate[sd]? to|connection between)",
    r"(list all|find all).+?(entities|relations|links|subsidiaries)",
]
_COMPLIANCE_PATTERNS = [
    r"(comply with|compliance with|violat|regulation requires|is it legal|is it allowed|are we allowed|gdpr require|hipaa|sox audit)",
    r"\b(permit|audit)\b.{0,30}\b(policy|regulation|law)\b",
]


def _heuristic_classify(query: str) -> QueryType | None:
    """Order: KG → Compliance → Simple.  Simple is last to avoid false positives."""
    q = query.lower()
    if any(re.search(p, q) for p in _KG_PATTERNS):
        return QueryType.KNOWLEDGE_GRAPH
    if any(re.search(p, q) for p in _COMPLIANCE_PATTERNS):
        return QueryType.COMPLIANCE
    if any(re.search(p, q) for p in _SIMPLE_PATTERNS):
        return QueryType.SIMPLE_RETRIEVAL
    return None


class QueryPlanner:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=_settings.openai_api_key, base_url=_settings.openai_base_url)

    async def plan(self, query: str, trace_id: str = "") -> PlannerDecision:
        async with trace_span("planner", trace_id):

            # Try fast heuristic first
            heuristic = _heuristic_classify(query)
            if heuristic in (QueryType.SIMPLE_RETRIEVAL,):
                decision = PlannerDecision(
                    query_type=QueryType.SIMPLE_RETRIEVAL,
                    use_hybrid_retrieval=True,
                    use_knowledge_graph=False,
                    verification_depth="minimal",
                    max_hops=1,
                    reasoning="Heuristic: simple factual lookup pattern.",
                )
                _log.info("planner_heuristic", query_type=decision.query_type, query=query)
                return decision

            # Fall back to LLM classification
            try:
                resp = await self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user",   "content": query},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content or "{}")
            except Exception as exc:
                _log.warning("planner_llm_failed", error=str(exc), query=query)
                raw = {}

            qt_str = raw.get("query_type", "simple_retrieval")
            try:
                qt = QueryType(qt_str)
            except ValueError:
                qt = QueryType.SIMPLE_RETRIEVAL

            decision = PlannerDecision(
                query_type=qt,
                use_hybrid_retrieval=bool(raw.get("use_hybrid_retrieval", True)),
                use_knowledge_graph=bool(raw.get("use_knowledge_graph", False)),
                verification_depth=raw.get("verification_depth", "standard"),
                max_hops=int(raw.get("max_hops", 1)),
                reasoning=raw.get("reasoning", ""),
            )

            _log.info(
                "planner_decision",
                trace_id=trace_id,
                query_type=decision.query_type,
                use_kg=decision.use_knowledge_graph,
                verification=decision.verification_depth,
                query=query,
            )
            return decision

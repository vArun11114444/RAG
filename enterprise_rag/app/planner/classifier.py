"""
Query classifier — Phase 4.

Classifies incoming queries into one of five types:
  SIMPLE       → single fact, direct answer
  MULTI_HOP    → requires chaining multiple facts
  KNOWLEDGE_GRAPH → entity relationships, graph traversal needed
  COMPLIANCE   → regulatory / policy queries needing strict verification
  RESEARCH     → open-ended synthesis across many sources

Uses keyword heuristics first (fast, no LLM cost), then LLM for ambiguous cases.
"""
from __future__ import annotations

import json
import re

from openai import AsyncOpenAI

from app.config import get_settings
from app.models.schemas import QueryType
from app.observability import get_logger, ERRORS

log = get_logger(__name__)
settings = get_settings()

# ── Heuristic rules ────────────────────────────────────────────────────────────

_COMPLIANCE_PATTERNS = re.compile(
    r"\b(regulation|compliance|gdpr|hipaa|sox|iso\s*\d+|policy|legal|"
    r"requirement|obligation|mandate|audit|certif|law\b|statute|directive)\b",
    re.IGNORECASE,
)

_GRAPH_PATTERNS = re.compile(
    r"\b(relationship|connected|related to|link between|how does .+ affect|"
    r"who does .+ work with|network of|influence|hierarchy)\b",
    re.IGNORECASE,
)

_MULTI_HOP_PATTERNS = re.compile(
    r"\b(why|explain how|step by step|process of|chain of|sequence|"
    r"cause.*effect|lead to|because of|as a result)\b",
    re.IGNORECASE,
)

_RESEARCH_PATTERNS = re.compile(
    r"\b(compare|contrast|overview of|summarise|summarize|literature|"
    r"all|every|comprehensive|across|between .+ and .+|pros and cons)\b",
    re.IGNORECASE,
)

_CLASSIFY_PROMPT = """\
Classify the following search query into exactly ONE category:
  simple          - single factual lookup, short answer
  multi_hop       - needs chaining multiple pieces of information
  knowledge_graph - about entity relationships or connections
  compliance      - regulatory, legal, or policy question
  research        - broad synthesis across many sources

Return ONLY a JSON object: {{"type": "<category>"}}

Query: {query}
"""


class QueryClassifier:
    """
    Two-stage query classifier:
      Stage 1 — regex heuristics (microseconds, free)
      Stage 2 — LLM disambiguation (when heuristics are uncertain)
    """

    def __init__(self) -> None:
        self._oai: AsyncOpenAI | None = None
        if settings.OPENAI_API_KEY:
            self._oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)

    def _heuristic_classify(self, query: str) -> QueryType | None:
        """
        Returns a QueryType if heuristics are confident, else None.
        Confidence = at least 2 pattern matches OR 1 very strong match.
        """
        scores: dict[QueryType, int] = {
            QueryType.COMPLIANCE: 0,
            QueryType.KNOWLEDGE_GRAPH: 0,
            QueryType.MULTI_HOP: 0,
            QueryType.RESEARCH: 0,
        }

        if _COMPLIANCE_PATTERNS.search(query):
            scores[QueryType.COMPLIANCE] += 2
        if _GRAPH_PATTERNS.search(query):
            scores[QueryType.KNOWLEDGE_GRAPH] += 2
        if _MULTI_HOP_PATTERNS.search(query):
            scores[QueryType.MULTI_HOP] += 1
        if _RESEARCH_PATTERNS.search(query):
            scores[QueryType.RESEARCH] += 1

        # Word count heuristic
        word_count = len(query.split())
        if word_count < 8:
            scores[QueryType.MULTI_HOP] = max(0, scores[QueryType.MULTI_HOP] - 1)

        best = max(scores, key=lambda k: scores[k])
        if scores[best] >= 2:
            return best
        if all(s == 0 for s in scores.values()):
            return QueryType.SIMPLE   # short, simple query
        return None   # ambiguous — escalate to LLM

    async def classify(self, query: str) -> QueryType:
        """
        Classify a query. Returns QueryType enum value.
        """
        # Fast path
        heuristic = self._heuristic_classify(query)
        if heuristic is not None:
            log.debug("Heuristic classification", extra={"query": query, "type": heuristic})
            return heuristic

        # LLM path
        if self._oai is None:
            log.warning("No LLM available for classification — defaulting to SIMPLE")
            return QueryType.SIMPLE

        try:
            resp = await self._oai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.0,
                max_tokens=32,
                messages=[
                    {"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}
                ],
            )
            raw = resp.choices[0].message.content or "{}"
            data = json.loads(raw)
            type_str = data.get("type", "simple").lower()
            result = QueryType(type_str)
            log.debug("LLM classification", extra={"query": query, "type": result})
            return result
        except Exception as exc:
            ERRORS.labels(phase="query_classifier", error_type=type(exc).__name__).inc()
            log.warning("LLM classification failed — defaulting to SIMPLE", extra={"error": str(exc)})
            return QueryType.SIMPLE

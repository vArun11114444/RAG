"""
app/services/retrieval/query_expander.py
LLM-backed query expansion – generates N alternative phrasings + sub-questions.
Falls back gracefully if the LLM call fails.
"""
from __future__ import annotations
import json

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.services.observability.logger import get_logger

_settings = get_settings()
_log = get_logger(__name__)

_SYSTEM = """You are a query expansion assistant for a document retrieval system.
Given a user query, generate 3 alternative phrasings that capture the same intent
plus 2 specific sub-questions that would help answer the original query comprehensively.
Respond ONLY with a JSON array of strings (no explanation).
Example output: ["alt query 1","alt query 2","alt query 3","sub-q 1","sub-q 2"]"""


class QueryExpander:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=_settings.openai_api_key, base_url=_settings.openai_base_url)

    async def expand(self, query: str) -> list[str]:
        """Return [original] + expanded alternatives.  Never raises."""
        try:
            resp = await self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or "[]"
            expansions: list[str] = json.loads(raw)
            _log.debug("query_expanded", original=query, n=len(expansions))
            return [query] + [q for q in expansions if isinstance(q, str)]
        except Exception as exc:
            _log.warning("query_expansion_failed", error=str(exc), query=query)
            return [query]

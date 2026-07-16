"""
Query expansion — generates semantically related query variants via LLM.
MIGRATION: Uses OPENAI_BASE_URL so OpenRouter works as the LLM backend.
"""
from __future__ import annotations
import asyncio
import json
from openai import AsyncOpenAI
from app.config import get_settings
from app.observability import get_logger

log = get_logger(__name__)
settings = get_settings()

_SYSTEM_PROMPT = """You are a query expansion assistant for a document retrieval system.
Given a user query, generate {n} alternative phrasings that capture the same intent
using different vocabulary, synonyms, and perspectives.
Return ONLY a JSON array of strings, no explanation.
Example: ["alt query 1", "alt query 2", "alt query 3"]"""


class QueryExpander:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,          # OpenRouter support
        )

    async def expand(self, query: str, n: int | None = None) -> list[str]:
        max_variants = n or settings.QUERY_EXPANSION_MAX
        if not settings.OPENAI_API_KEY:
            return [query]
        try:
            resp = await self._client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT.format(n=max_variants)},
                    {"role": "user",   "content": query},
                ],
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or "[]"
            variants: list[str] = json.loads(raw)
            if not isinstance(variants, list):
                raise ValueError("Unexpected format")
            seen = {query}
            result = [query]
            for v in variants:
                if isinstance(v, str) and v not in seen:
                    result.append(v); seen.add(v)
            log.debug("Query expanded", extra={"original": query, "variants": result[1:]})
            return result
        except Exception as exc:
            log.warning("Query expansion failed", extra={"error": str(exc)})
            return [query]

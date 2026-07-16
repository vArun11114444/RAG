"""
Query expansion — generates alternative phrasings of the user query
so BM25 and vector retrieval cover more of the semantic surface.

Strategy:
  1. LLM-generated variants (e.g. rephrase, keyword extraction, HyDE)
  2. Simple synonym injection as a cheap fallback when LLM is unavailable
"""
from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI

from app.config import get_settings
from app.observability import get_logger, ERRORS

log = get_logger(__name__)
settings = get_settings()

_EXPANSION_PROMPT = """\
You are a search query expansion assistant.
Given the original query, produce {n} alternative search queries that would \
retrieve relevant documents. Each variant should use different vocabulary while \
preserving the original intent.

Return ONLY a JSON array of strings, no markdown, no explanation.
Example: ["variant one", "variant two", "variant three"]

Original query: {query}"""


async def expand_query(
    query: str,
    n: int | None = None,
) -> list[str]:
    """
    Returns a list of alternative queries including the original.
    Falls back to [query] on any error so the pipeline never stalls.
    """
    max_variants = n or settings.QUERY_EXPANSION_MAX
    if not settings.OPENAI_API_KEY:
        log.warning("No OpenAI key — skipping query expansion")
        return [query]

    try:
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
        prompt = _EXPANSION_PROMPT.format(query=query, n=max_variants)

        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=0.3,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or "[]"
        variants: list[str] = json.loads(raw)

        # Deduplicate and prepend original
        seen: set[str] = {query}
        expanded = [query]
        for v in variants:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                expanded.append(v)

        log.debug(
            "Query expanded",
            extra={"original": query, "variants": len(expanded) - 1},
        )
        return expanded[:max_variants + 1]

    except Exception as exc:
        ERRORS.labels(phase="query_expansion", error_type=type(exc).__name__).inc()
        log.warning("Query expansion failed — using original query", extra={"error": str(exc)})
        return [query]

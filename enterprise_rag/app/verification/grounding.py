"""
Source grounding and citation validation — Phase 3.

Grounding: checks that each claim in the generated answer is
           anchored in at least one retrieved chunk.
Citation:  validates that citation references point to real chunks
           and that the cited text supports the claim.
"""
from __future__ import annotations

import re
import uuid
from difflib import SequenceMatcher
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.models.schemas import CitationValidation, RetrievedChunk
from app.observability import get_logger, ERRORS

log = get_logger(__name__)
settings = get_settings()

_GROUNDING_PROMPT = """\
You are a fact-checking assistant. Given an answer and a set of source passages,
identify each factual claim in the answer and determine which source passage
(if any) supports it.

Return ONLY a JSON array of objects with keys:
  claim       - the exact sentence or phrase from the answer
  supported   - true/false
  source_idx  - index into sources array (0-based), or null if not supported
  confidence  - float 0.0-1.0

Answer:
{answer}

Sources (indexed):
{sources}
"""


def _text_overlap(a: str, b: str) -> float:
    """Return ratio of overlapping tokens between two strings."""
    return SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


class GroundingValidator:
    """
    Validates that generated answer text is grounded in retrieved chunks.
    """

    def __init__(self) -> None:
        self._oai: AsyncOpenAI | None = None
        if settings.OPENAI_API_KEY:
            self._oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)

    async def validate(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[CitationValidation]:
        """
        Returns a CitationValidation per claim found in the answer.
        Falls back to lexical overlap if LLM unavailable.
        """
        if not chunks:
            return []

        if self._oai:
            return await self._llm_grounding(answer, chunks)
        return self._lexical_grounding(answer, chunks)

    async def _llm_grounding(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[CitationValidation]:
        sources_str = "\n\n".join(
            f"[{i}] ({c.source}) {c.text[:400]}"
            for i, c in enumerate(chunks)
        )
        prompt = _GROUNDING_PROMPT.format(answer=answer, sources=sources_str)

        try:
            import json
            resp = await self._oai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=0.0,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content or "[]"
            items: list[dict[str, Any]] = json.loads(raw)

            validations: list[CitationValidation] = []
            for item in items:
                src_idx = item.get("source_idx")
                chunk = chunks[src_idx] if src_idx is not None else chunks[0]
                validations.append(
                    CitationValidation(
                        citation_id=str(uuid.uuid4()),
                        is_valid=bool(item.get("supported", False)),
                        source_chunk_id=chunk.chunk_id,
                        claim_text=str(item.get("claim", "")),
                        supporting_text=chunk.text[:200],
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )
            return validations

        except Exception as exc:
            ERRORS.labels(phase="grounding", error_type=type(exc).__name__).inc()
            log.warning("LLM grounding failed — falling back", extra={"error": str(exc)})
            return self._lexical_grounding(answer, chunks)

    def _lexical_grounding(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[CitationValidation]:
        """Simple sentence-level lexical overlap grounding."""
        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        validations: list[CitationValidation] = []

        for sentence in sentences:
            best_score = 0.0
            best_chunk = chunks[0]
            for chunk in chunks:
                score = _text_overlap(sentence, chunk.text)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk

            validations.append(
                CitationValidation(
                    citation_id=str(uuid.uuid4()),
                    is_valid=best_score >= 0.15,
                    source_chunk_id=best_chunk.chunk_id,
                    claim_text=sentence,
                    supporting_text=best_chunk.text[:200],
                    confidence=round(best_score, 3),
                )
            )

        return validations

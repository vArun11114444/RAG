"""
Contradiction detection — Phase 3.

Detects when two retrieved chunks make conflicting factual claims.
Uses an NLI cross-encoder (deberta) if available, otherwise
falls back to a lightweight LLM-based check.
"""
from __future__ import annotations

import asyncio
import itertools
import uuid
from typing import Any

from app.config import get_settings
from app.models.schemas import ContradictionPair, RetrievedChunk
from app.observability import get_logger, ERRORS, CONTRADICTIONS_FOUND

log = get_logger(__name__)
settings = get_settings()

# Labels returned by cross-encoder NLI models
_CONTRADICTION_LABEL = "contradiction"


class ContradictionDetector:
    """
    Detects contradictory chunk pairs via NLI or LLM.
    Only runs pairwise checks on the top-N chunks to keep latency bounded.
    """

    MAX_PAIRS = 15   # cap combinatorial explosion

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._oai: Any = None
        self._init_nli()

    def _init_nli(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._pipeline = CrossEncoder(
                settings.HALLUCINATION_NLI_MODEL,
                max_length=512,
            )
            log.info("NLI cross-encoder loaded", extra={"model": settings.HALLUCINATION_NLI_MODEL})
        except Exception as exc:
            log.warning(
                "NLI model unavailable — contradiction detection degraded",
                extra={"reason": str(exc)},
            )

        if settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            self._oai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)

    async def detect(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[ContradictionPair]:
        """
        Check chunk pairs for contradictions.
        Returns list of detected ContradictionPair objects.
        """
        if len(chunks) < 2:
            return []

        # Build candidate pairs (cap at MAX_PAIRS)
        pairs = list(itertools.combinations(chunks, 2))[: self.MAX_PAIRS]

        if self._pipeline is not None:
            return await self._nli_detect(pairs)
        if self._oai is not None:
            return await self._llm_detect(pairs)

        log.warning("No contradiction backend available")
        return []

    async def _nli_detect(
        self, pairs: list[tuple[RetrievedChunk, RetrievedChunk]]
    ) -> list[ContradictionPair]:
        """Run sentence-transformers CrossEncoder on chunk pairs."""
        def _score_pairs() -> list[dict[str, Any]]:
            sentence_pairs = [
                (a.text[:512], b.text[:512]) for a, b in pairs
            ]
            scores = self._pipeline.predict(
                sentence_pairs,
                apply_softmax=True,
            )
            results = []
            for (a, b), score_vec in zip(pairs, scores):
                # score_vec: [entailment, neutral, contradiction]
                labels = self._pipeline.config.id2label
                label_scores = {labels[i]: float(s) for i, s in enumerate(score_vec)}
                contradiction_score = label_scores.get(_CONTRADICTION_LABEL, 0.0)
                results.append({
                    "chunk_a": a,
                    "chunk_b": b,
                    "score": contradiction_score,
                })
            return results

        try:
            scored = await asyncio.to_thread(_score_pairs)
            return self._build_pairs(scored)
        except Exception as exc:
            ERRORS.labels(phase="contradiction_nli", error_type=type(exc).__name__).inc()
            log.warning("NLI scoring failed", extra={"error": str(exc)})
            return []

    async def _llm_detect(
        self, pairs: list[tuple[RetrievedChunk, RetrievedChunk]]
    ) -> list[ContradictionPair]:
        """Use LLM as fallback contradiction detector."""
        import json

        async def _check_pair(a: RetrievedChunk, b: RetrievedChunk) -> dict[str, Any]:
            prompt = (
                "Do these two passages contradict each other?\n\n"
                f"Passage A: {a.text[:400]}\n\nPassage B: {b.text[:400]}\n\n"
                'Reply ONLY with JSON: {"contradicts": true/false, "score": 0.0-1.0}'
            )
            try:
                resp = await self._oai.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    temperature=0.0,
                    max_tokens=64,
                    messages=[{"role": "user", "content": prompt}],
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                return {"chunk_a": a, "chunk_b": b, "score": float(data.get("score", 0.0))}
            except Exception:
                return {"chunk_a": a, "chunk_b": b, "score": 0.0}

        scored = await asyncio.gather(*[_check_pair(a, b) for a, b in pairs])
        return self._build_pairs(list(scored))

    def _build_pairs(
        self, scored: list[dict[str, Any]]
    ) -> list[ContradictionPair]:
        found: list[ContradictionPair] = []
        threshold = settings.CONTRADICTION_THRESHOLD
        for item in scored:
            if item["score"] >= threshold:
                a: RetrievedChunk = item["chunk_a"]
                b: RetrievedChunk = item["chunk_b"]
                found.append(
                    ContradictionPair(
                        chunk_id_a=a.chunk_id,
                        chunk_id_b=b.chunk_id,
                        claim_a=a.text[:200],
                        claim_b=b.text[:200],
                        contradiction_score=item["score"],
                    )
                )
                CONTRADICTIONS_FOUND.inc()
                log.warning(
                    "Contradiction detected",
                    extra={
                        "chunk_a": a.chunk_id,
                        "chunk_b": b.chunk_id,
                        "score": item["score"],
                    },
                )
        return found

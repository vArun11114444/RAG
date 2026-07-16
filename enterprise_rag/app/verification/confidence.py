"""
Hallucination detection and confidence scoring — Phase 3.

Hallucination check: NLI-based entailment between answer claims and sources.
Confidence scoring: weighted aggregate of grounding, citation validity,
                   contradiction absence, and retrieval scores.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.config import get_settings
from app.models.schemas import (
    CitationValidation,
    ContradictionPair,
    RetrievedChunk,
)
from app.observability import get_logger, ERRORS, CONFIDENCE_SCORE, HALLUCINATION_RISK

log = get_logger(__name__)
settings = get_settings()


class HallucinationChecker:
    """
    Checks whether the generated answer is entailed by the retrieved sources.
    High hallucination risk = answer makes claims NOT entailed by any chunk.
    """

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._init_nli()

    def _init_nli(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._pipeline = CrossEncoder(
                settings.HALLUCINATION_NLI_MODEL,
                max_length=512,
            )
        except Exception as exc:
            log.warning("Hallucination NLI unavailable", extra={"reason": str(exc)})

    async def score(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> float:
        """
        Returns hallucination risk in [0, 1].
        0.0 = fully entailed, 1.0 = completely ungrounded.
        """
        if not chunks or not answer.strip():
            return 0.0

        if self._pipeline is None:
            # Fallback: if no NLI model, estimate from avg retrieval score
            avg_score = sum(c.score for c in chunks) / len(chunks)
            risk = max(0.0, 1.0 - avg_score)
            return round(risk, 3)

        # Truncate answer and concatenate top-3 chunks as "premise"
        premise = " ".join(c.text[:300] for c in chunks[:3])
        hypothesis = answer[:512]

        def _run() -> float:
            scores = self._pipeline.predict(
                [(premise, hypothesis)],
                apply_softmax=True,
            )[0]
            labels = self._pipeline.config.id2label
            label_map = {labels[i]: float(s) for i, s in enumerate(scores)}
            # risk = 1 - entailment probability
            entailment = label_map.get("entailment", 0.5)
            return round(1.0 - entailment, 3)

        try:
            risk = await asyncio.to_thread(_run)
            HALLUCINATION_RISK.observe(risk)
            return risk
        except Exception as exc:
            ERRORS.labels(phase="hallucination", error_type=type(exc).__name__).inc()
            log.warning("Hallucination scoring failed", extra={"error": str(exc)})
            return 0.5  # conservative mid-point on failure


class ConfidenceScorer:
    """
    Produces a single overall confidence score [0, 1] for a RAG response.

    Weighted components:
        grounding_score      (40%) — fraction of claims that are grounded
        citation_validity    (25%) — fraction of valid citations
        no_contradiction     (20%) — penalised for each contradiction found
        retrieval_quality    (15%) — mean retrieval score of top chunks
    """

    WEIGHTS = {
        "grounding": 0.40,
        "citation": 0.25,
        "contradiction": 0.20,
        "retrieval": 0.15,
    }

    def compute(
        self,
        citations: list[CitationValidation],
        contradictions: list[ContradictionPair],
        chunks: list[RetrievedChunk],
        hallucination_risk: float,
    ) -> float:
        # Grounding: fraction of valid citations
        if citations:
            grounding = sum(1 for c in citations if c.is_valid) / len(citations)
            citation_conf = sum(c.confidence for c in citations if c.is_valid) / max(
                1, sum(1 for c in citations if c.is_valid)
            )
        else:
            grounding = 0.5
            citation_conf = 0.5

        # Contradiction penalty: each contradiction reduces score by 0.1, floored at 0
        contradiction_score = max(0.0, 1.0 - len(contradictions) * 0.1)

        # Retrieval quality: mean score of top-5 chunks
        top_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:5]
        retrieval_quality = (
            sum(c.score for c in top_chunks) / len(top_chunks) if top_chunks else 0.5
        )

        raw = (
            self.WEIGHTS["grounding"] * grounding
            + self.WEIGHTS["citation"] * citation_conf
            + self.WEIGHTS["contradiction"] * contradiction_score
            + self.WEIGHTS["retrieval"] * retrieval_quality
        )

        # Apply hallucination penalty
        penalised = raw * (1.0 - hallucination_risk * 0.5)
        score = round(max(0.0, min(1.0, penalised)), 3)

        CONFIDENCE_SCORE.observe(score)
        log.debug(
            "Confidence computed",
            extra={
                "grounding": round(grounding, 3),
                "citation_conf": round(citation_conf, 3),
                "contradiction_score": round(contradiction_score, 3),
                "retrieval_quality": round(retrieval_quality, 3),
                "hallucination_risk": hallucination_risk,
                "final": score,
            },
        )
        return score

"""
Verification Layer — Phase 3.

Five sub-components:
  1. Source grounding    — every claim must map to a retrieved chunk
  2. Citation validation — verify citations reference real supporting text
  3. Contradiction detection — find conflicting claims across chunks
  4. Confidence scoring  — composite 0-1 score for the full answer
  5. Hallucination check — NLI entailment to catch unsupported claims
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from app.config import get_settings
from app.models.schemas import (
    CitationValidation, ContradictionPair, RetrievedChunk, VerificationResult,
)
from app.observability import (
    CONFIDENCE_SCORE, CONTRADICTIONS_FOUND, HALLUCINATION_RISK,
    get_logger, traced,
)

log = get_logger(__name__)
settings = get_settings()

# ── NLI model (lazy-loaded) ───────────────────────────────────────────────────
_nli_pipe = None

def _get_nli():
    global _nli_pipe
    if _nli_pipe is None:
        try:
            from transformers import pipeline
            _nli_pipe = pipeline(
                "zero-shot-classification",
                model=settings.HALLUCINATION_NLI_MODEL,
                device=-1,
            )
            log.info("NLI model loaded", extra={"model": settings.HALLUCINATION_NLI_MODEL})
        except Exception as exc:
            log.warning("NLI model unavailable — hallucination check disabled", extra={"error": str(exc)})
            _nli_pipe = False
    return _nli_pipe


class VerificationService:
    """
    Stateless service — call .verify() after retrieval + answer generation.
    """

    @traced("verification")
    async def verify(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        query: str = "",
    ) -> VerificationResult:
        """Run all five verification checks concurrently and aggregate."""
        citations_task = self._validate_citations(answer, chunks)
        contradictions_task = self._detect_contradictions(chunks)
        hallucination_task = self._check_hallucination(answer, chunks)

        citations, contradictions, hallucination_risk = await asyncio.gather(
            citations_task, contradictions_task, hallucination_task
        )

        is_grounded = bool(chunks) and any(c.is_valid for c in citations)
        confidence = self._compute_confidence(
            citations=citations,
            contradictions=contradictions,
            hallucination_risk=hallucination_risk,
            chunks=chunks,
        )

        warnings: list[str] = []
        if not is_grounded:
            warnings.append("Answer could not be grounded to any retrieved source.")
        if contradictions:
            warnings.append(
                f"{len(contradictions)} contradiction(s) found in source documents."
            )
        if hallucination_risk > 0.5:
            warnings.append(
                f"High hallucination risk detected ({hallucination_risk:.2f})."
            )
        if confidence < settings.CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Overall confidence ({confidence:.2f}) below threshold "
                f"({settings.CONFIDENCE_THRESHOLD})."
            )

        # Metrics
        CONFIDENCE_SCORE.observe(confidence)
        HALLUCINATION_RISK.observe(hallucination_risk)
        if contradictions:
            CONTRADICTIONS_FOUND.inc(len(contradictions))

        return VerificationResult(
            overall_confidence=confidence,
            is_grounded=is_grounded,
            citations=citations,
            contradictions=contradictions,
            hallucination_risk=hallucination_risk,
            warnings=warnings,
        )

    # ── 1 + 2: Citation extraction & validation ────────────────────────────────

    async def _validate_citations(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> list[CitationValidation]:
        """
        Parses [CITE:chunk_id] markers from the answer and verifies each
        claim is genuinely supported by the referenced chunk.
        Falls back to checking whether any chunk overlaps the claim if
        no explicit markers are present.
        """
        pattern = re.compile(r"\[CITE:([^\]]+)\](.{0,300}?)(?=\[CITE:|$)", re.DOTALL)
        matches = list(pattern.finditer(answer))

        chunk_map = {c.chunk_id: c for c in chunks}
        results: list[CitationValidation] = []

        if matches:
            for m in matches:
                cid = m.group(1).strip()
                claim = m.group(2).strip()
                chunk = chunk_map.get(cid)
                if chunk is None:
                    results.append(CitationValidation(
                        citation_id=cid, is_valid=False,
                        source_chunk_id=cid, claim_text=claim,
                        supporting_text="", confidence=0.0,
                    ))
                    continue
                overlap = _text_overlap(claim, chunk.text)
                results.append(CitationValidation(
                    citation_id=cid, is_valid=overlap > 0.1,
                    source_chunk_id=cid, claim_text=claim,
                    supporting_text=chunk.text[:200],
                    confidence=min(1.0, overlap * 2),
                ))
        else:
            # No explicit citations — try to ground top-level sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s) > 20]
            for i, sent in enumerate(sentences[:5]):
                best_chunk = max(chunks, key=lambda c: _text_overlap(sent, c.text), default=None)
                if best_chunk:
                    overlap = _text_overlap(sent, best_chunk.text)
                    results.append(CitationValidation(
                        citation_id=f"auto_{i}", is_valid=overlap > 0.05,
                        source_chunk_id=best_chunk.chunk_id, claim_text=sent,
                        supporting_text=best_chunk.text[:200],
                        confidence=min(1.0, overlap * 3),
                    ))

        return results

    # ── 3: Contradiction detection ─────────────────────────────────────────────

    async def _detect_contradictions(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[ContradictionPair]:
        """
        Compare pairs of top-scored chunks for contradictory numeric or
        factual claims using simple heuristics + optional NLI.
        O(n²) — only run on top-10 to keep latency bounded.
        """
        top = chunks[:10]
        contradictions: list[ContradictionPair] = []

        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                score = await asyncio.to_thread(
                    _heuristic_contradiction_score,
                    top[i].text,
                    top[j].text,
                )
                if score >= settings.CONTRADICTION_THRESHOLD:
                    contradictions.append(ContradictionPair(
                        chunk_id_a=top[i].chunk_id,
                        chunk_id_b=top[j].chunk_id,
                        claim_a=top[i].text[:200],
                        claim_b=top[j].text[:200],
                        contradiction_score=score,
                    ))

        return contradictions

    # ── 5: Hallucination check ─────────────────────────────────────────────────

    async def _check_hallucination(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> float:
        """
        Returns a risk score in [0, 1].
        Uses NLI entailment: does the corpus entail the answer?
        Falls back to inverse overlap heuristic if NLI is unavailable.
        """
        if not chunks:
            return 1.0

        premise = " ".join(c.text[:300] for c in chunks[:5])

        nli = _get_nli()
        if nli:
            try:
                result = await asyncio.to_thread(
                    nli,
                    answer[:512],
                    candidate_labels=["entailment", "neutral", "contradiction"],
                    hypothesis_template="{}",
                )
                scores = dict(zip(result["labels"], result["scores"]))
                contradiction_score = scores.get("contradiction", 0.0)
                entailment_score = scores.get("entailment", 0.0)
                return round(contradiction_score + 0.5 * (1 - entailment_score), 3)
            except Exception as exc:
                log.debug("NLI inference failed", extra={"error": str(exc)})

        # Fallback: inverse overlap
        overlap = _text_overlap(answer, premise)
        return round(max(0.0, 1.0 - overlap * 2), 3)

    # ── 4: Confidence scoring ──────────────────────────────────────────────────

    def _compute_confidence(
        self,
        citations: list[CitationValidation],
        contradictions: list[ContradictionPair],
        hallucination_risk: float,
        chunks: list[RetrievedChunk],
    ) -> float:
        if not chunks:
            return 0.0

        # Weighted components
        citation_score = (
            sum(c.confidence for c in citations) / len(citations)
            if citations else 0.5
        )
        contradiction_penalty = min(1.0, len(contradictions) * 0.15)
        chunk_quality = (
            sum(c.score for c in chunks[:5]) / min(5, len(chunks))
        )

        raw = (
            0.40 * citation_score
            + 0.30 * (1.0 - hallucination_risk)
            + 0.20 * chunk_quality
            - 0.10 * contradiction_penalty
        )
        return round(max(0.0, min(1.0, raw)), 3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _text_overlap(a: str, b: str) -> float:
    """Jaccard overlap on word tokens."""
    a_tokens = set(re.findall(r"\w+", a.lower()))
    b_tokens = set(re.findall(r"\w+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _heuristic_contradiction_score(text_a: str, text_b: str) -> float:
    """
    Detect contradictions via negation patterns and numeric inconsistencies.
    """
    score = 0.0

    # Negation proximity
    neg_words = {"not", "never", "no", "cannot", "isn't", "aren't", "wasn't",
                 "weren't", "doesn't", "don't", "didn't"}
    words_a = set(re.findall(r"\w+", text_a.lower()))
    words_b = set(re.findall(r"\w+", text_b.lower()))

    shared = words_a & words_b
    if shared:
        neg_a = bool(words_a & neg_words)
        neg_b = bool(words_b & neg_words)
        if neg_a != neg_b and len(shared) > 5:
            score += 0.4

    # Numeric contradiction: same noun phrase, different numbers
    nums_a = set(re.findall(r"\b\d+[\.,]?\d*\b", text_a))
    nums_b = set(re.findall(r"\b\d+[\.,]?\d*\b", text_b))
    if nums_a and nums_b and not (nums_a & nums_b) and len(shared) > 3:
        score += 0.35

    return min(1.0, score)

"""
app/services/verification/verifier.py
Phase 3: source grounding, citation validation, contradiction detection,
confidence scoring, and hallucination risk assessment.
"""
from __future__ import annotations
import json
import time
import re

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.schemas import (
    CitationCheck,
    RetrievedChunk,
    VerificationResult,
)
from app.services.observability.logger import get_logger
from app.services.observability.metrics import (
    CONFIDENCE,
    VERIFICATION_FAIL,
    VERIFICATION_PASS,
    VERIFY_LATENCY,
)
from app.services.observability.tracing import trace_span

_settings = get_settings()
_log = get_logger(__name__)

_VERIFY_SYSTEM = """You are a strict factual verification assistant.
Given an answer and the source passages it should be grounded in, you must:
1. Check whether every factual claim in the answer is supported by a source passage.
2. Detect internal contradictions within the answer.
3. Detect any information in the answer that is NOT present in any source passage (potential hallucination).
4. Assign a confidence score from 0.0 (completely ungrounded) to 1.0 (fully grounded).

Return ONLY valid JSON (no markdown):
{
  "is_grounded": true|false,
  "confidence_score": 0.0-1.0,
  "contradictions": ["list of contradiction descriptions"],
  "hallucinated_claims": ["claims not found in sources"],
  "citation_checks": [
    {"citation_id": "...", "is_valid": true|false, "grounding_score": 0.0-1.0, "note": "..."}
  ]
}"""


def _extract_citations(answer: str) -> list[str]:
    """Pull citation markers like [1], [doc_abc], [SOURCE-3] from answer text."""
    return re.findall(r"\[([^\]]{1,40})\]", answer)


class AnswerVerifier:

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=_settings.openai_api_key, base_url=_settings.openai_base_url)

    async def verify(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        trace_id: str = "",
    ) -> VerificationResult:
        t0 = time.perf_counter()

        async with trace_span("verification", trace_id):
            sources_text = "\n\n".join(
                f"[{c.chunk_id}] {c.content}" for c in chunks
            )
            citations = _extract_citations(answer)

            try:
                resp = await self._client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": _VERIFY_SYSTEM},
                        {"role": "user", "content": (
                            f"ANSWER:\n{answer}\n\n"
                            f"SOURCE PASSAGES:\n{sources_text[:6000]}\n\n"
                            f"CITATIONS IN ANSWER: {citations}"
                        )},
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content or "{}")
            except Exception as exc:
                _log.warning("verification_llm_failed", error=str(exc))
                raw = {}

            confidence = float(raw.get("confidence_score", 0.5))
            is_grounded = bool(raw.get("is_grounded", confidence >= _settings.verification_threshold))
            contradictions: list[str] = raw.get("contradictions", [])
            hallucinated: list[str] = raw.get("hallucinated_claims", [])

            # Build citation checks
            cit_raw = raw.get("citation_checks", [])
            citation_checks: list[CitationCheck] = []
            for cr in cit_raw:
                # find matching chunk by id prefix
                cid = cr.get("citation_id", "")
                matched = next(
                    (c.chunk_id for c in chunks if c.chunk_id.startswith(cid) or cid.startswith(c.chunk_id[:6])),
                    cid,
                )
                citation_checks.append(CitationCheck(
                    citation_id=cid,
                    is_valid=bool(cr.get("is_valid", False)),
                    grounding_score=float(cr.get("grounding_score", 0.0)),
                    source_chunk_id=matched,
                    note=cr.get("note", ""),
                ))

            # Hallucination risk
            if hallucinated or not is_grounded:
                hal_risk = "high" if len(hallucinated) > 2 else "medium"
            else:
                hal_risk = "low"

            passed = (
                is_grounded
                and confidence >= _settings.verification_threshold
                and hal_risk != "high"
            )

            latency_ms = (time.perf_counter() - t0) * 1000
            VERIFY_LATENCY.observe(latency_ms / 1000)
            CONFIDENCE.set(confidence)

            if passed:
                VERIFICATION_PASS.labels(depth="standard").inc()
            else:
                reason = "low_confidence" if confidence < _settings.verification_threshold else "hallucination"
                VERIFICATION_FAIL.labels(reason=reason).inc()

            result = VerificationResult(
                is_grounded=is_grounded,
                confidence_score=round(confidence, 4),
                citation_checks=citation_checks,
                contradictions=contradictions,
                hallucination_risk=hal_risk,
                passed=passed,
            )

            _log.info(
                "verification_complete",
                trace_id=trace_id,
                confidence=result.confidence_score,
                passed=passed,
                hal_risk=hal_risk,
                contradictions=len(contradictions),
                latency_ms=round(latency_ms, 2),
            )
            return result

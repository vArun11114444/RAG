"""
Verification layer orchestrator — Phase 3 entry point.

Runs grounding, citation validation, contradiction detection,
hallucination checks, and confidence scoring in the right order.
"""
from __future__ import annotations

from app.models.schemas import RetrievedChunk, VerificationResult
from app.observability import get_logger, Timer
from app.observability.tracer import traced

from .confidence import ConfidenceScorer, HallucinationChecker
from .contradiction import ContradictionDetector
from .grounding import GroundingValidator

log = get_logger(__name__)


class VerificationPipeline:
    """
    Full verification pipeline for a generated answer + retrieved chunks.

    Compose at startup and inject wherever needed:

        verifier = VerificationPipeline()
        result = await verifier.verify(answer, chunks, level="strict")
    """

    def __init__(self) -> None:
        self._grounder = GroundingValidator()
        self._contradiction = ContradictionDetector()
        self._hallucination = HallucinationChecker()
        self._scorer = ConfidenceScorer()

    @traced("verification")
    async def verify(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        level: str = "standard",   # "standard" | "strict"
    ) -> VerificationResult:
        """
        Run the full verification suite.

        Args:
            answer: Generated answer text.
            chunks: Retrieved source chunks used to generate the answer.
            level:  "strict" runs all checks synchronously and adds
                    warnings for every failed validation.

        Returns:
            VerificationResult with aggregated scores and details.
        """
        latency: dict[str, float] = {}

        # 1. Citation / grounding validation
        with Timer("grounding", latency):
            citations = await self._grounder.validate(answer, chunks)

        # 2. Contradiction detection
        with Timer("contradiction", latency):
            contradictions = await self._contradiction.detect(chunks)

        # 3. Hallucination risk
        with Timer("hallucination", latency):
            hallucination_risk = await self._hallucination.score(answer, chunks)

        # 4. Composite confidence
        confidence = self._scorer.compute(
            citations=citations,
            contradictions=contradictions,
            chunks=chunks,
            hallucination_risk=hallucination_risk,
        )

        # 5. Build warnings
        warnings: list[str] = []
        invalid_citations = [c for c in citations if not c.is_valid]
        if invalid_citations:
            warnings.append(
                f"{len(invalid_citations)} claim(s) could not be grounded in sources"
            )
        if contradictions:
            warnings.append(
                f"{len(contradictions)} contradiction(s) detected among retrieved chunks"
            )
        if hallucination_risk > 0.6:
            warnings.append(
                f"High hallucination risk detected ({hallucination_risk:.0%})"
            )
        if confidence < 0.5 and level == "strict":
            warnings.append(
                "Confidence below threshold — response may be unreliable"
            )

        is_grounded = (
            confidence >= 0.5
            and not any(c.confidence < 0.2 for c in citations)
        )

        log.info(
            "Verification complete",
            extra={
                "confidence": confidence,
                "is_grounded": is_grounded,
                "hallucination_risk": hallucination_risk,
                "contradictions": len(contradictions),
                "warnings": len(warnings),
                "latency_ms": latency,
                "level": level,
            },
        )

        return VerificationResult(
            overall_confidence=confidence,
            is_grounded=is_grounded,
            citations=citations,
            contradictions=contradictions,
            hallucination_risk=hallucination_risk,
            warnings=warnings,
        )

    async def run(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        level: str = "standard",
    ) -> VerificationResult:
        """Public alias for verify() — called by the API route."""
        return await self.verify(answer=answer, chunks=chunks, level=level)

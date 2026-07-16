"""
Tests for Phase 3 — Verification Layer.
"""
import pytest
from app.models.schemas import RetrievedChunk
from app.verification.grounding import GroundingValidator
from app.verification.contradiction import ContradictionDetector
from app.verification.confidence import ConfidenceScorer, HallucinationChecker
from app.verification.pipeline import VerificationPipeline


def _chunk(cid: str, text: str, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, document_id="doc1", text=text,
        score=score, source="test.pdf", retrieval_method="vector",
    )


@pytest.mark.asyncio
async def test_grounding_lexical_fallback():
    validator = GroundingValidator()
    chunks = [_chunk("c1", "The Eiffel Tower is located in Paris, France.")]
    answer = "The Eiffel Tower stands in Paris."
    results = await validator.validate(answer, chunks)
    assert len(results) > 0
    assert any(r.is_valid for r in results)


@pytest.mark.asyncio
async def test_grounding_empty_chunks():
    validator = GroundingValidator()
    results = await validator.validate("Some answer", [])
    assert results == []


@pytest.mark.asyncio
async def test_contradiction_not_enough_chunks():
    detector = ContradictionDetector()
    results = await detector.detect([_chunk("c1", "Sky is blue.")])
    assert results == []


def test_confidence_scorer_basic():
    from app.models.schemas import CitationValidation, ContradictionPair
    scorer = ConfidenceScorer()
    citations = [
        CitationValidation(citation_id="1", is_valid=True, source_chunk_id="c1",
                           claim_text="claim", supporting_text="support", confidence=0.9),
    ]
    score = scorer.compute(
        citations=citations, contradictions=[], 
        chunks=[_chunk("c1", "text", score=0.85)], hallucination_risk=0.1,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.5


@pytest.mark.asyncio
async def test_hallucination_fallback_no_model():
    checker = HallucinationChecker()
    checker._pipeline = None  # Force fallback
    chunks = [_chunk("c1", "Paris is the capital of France.", score=0.9)]
    risk = await checker.score("Paris is the capital of France.", chunks)
    assert 0.0 <= risk <= 1.0


@pytest.mark.asyncio
async def test_verification_pipeline_standard():
    pipeline = VerificationPipeline()
    chunks = [
        _chunk("c1", "Machine learning is a subset of artificial intelligence."),
        _chunk("c2", "Deep learning uses neural networks with many layers."),
    ]
    answer = "Machine learning is part of AI. Deep learning uses neural networks."
    result = await pipeline.verify(answer=answer, chunks=chunks, level="standard")
    assert result is not None
    assert 0.0 <= result.overall_confidence <= 1.0
    assert isinstance(result.is_grounded, bool)
    assert isinstance(result.warnings, list)

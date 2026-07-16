"""Tests for RRF, BM25, and score normalization — Phase 1."""
import pytest
from app.hybrid.rrf import reciprocal_rank_fusion, normalize_scores
from app.models.schemas import RetrievedChunk


def chunk(cid, score=1.0, method="bm25"):
    return RetrievedChunk(
        chunk_id=cid, document_id="doc", text="text",
        score=score, source="src", retrieval_method=method,
    )


def test_rrf_single_list():
    lst = [chunk("a", 0.9), chunk("b", 0.7), chunk("c", 0.5)]
    fused, scores = reciprocal_rank_fusion(lst)
    assert [c.chunk_id for c in fused] == ["a", "b", "c"]


def test_rrf_overlap_boosts_rank():
    a = [chunk("x", 0.9), chunk("y", 0.7)]
    b = [chunk("y", 0.95), chunk("z", 0.6)]
    fused, _ = reciprocal_rank_fusion(a, b)
    ids = [c.chunk_id for c in fused]
    assert "y" in ids
    assert ids.index("y") == 0  # y in both lists → top


def test_normalize_scores_range():
    chunks = [chunk("a", 10), chunk("b", 5), chunk("c", 0)]
    result = normalize_scores(chunks)
    scores = [c.score for c in result]
    assert max(scores) == pytest.approx(1.0)
    assert min(scores) == pytest.approx(0.0)


def test_normalize_single_chunk():
    chunks = [chunk("a", 7.5)]
    result = normalize_scores(chunks)
    assert result[0].score == pytest.approx(1.0)


def test_rrf_top_k_respected():
    lists = [[chunk(str(i), 1.0 - i * 0.05) for i in range(20)]]
    fused, _ = reciprocal_rank_fusion(*lists, top_k=5)
    assert len(fused) <= 5

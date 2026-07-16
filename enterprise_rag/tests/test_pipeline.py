"""
Integration tests for the full pipeline — mocks external dependencies.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.models.schemas import (
    RAGRequest, QueryType, RetrievedChunk, HybridRetrievalResult,
    QueryPlan, RetrievalStrategy, VerificationResult,
)


def _make_chunk(cid="c1"):
    return RetrievedChunk(
        chunk_id=cid, document_id="doc1",
        text="Artificial intelligence is transforming industries worldwide.",
        score=0.9, source="test.pdf", retrieval_method="fused",
    )


def _make_plan(qtype=QueryType.SIMPLE):
    return QueryPlan(
        query_type=qtype,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        use_graph=False, require_verification=False,
        verification_level="standard", max_hops=1,
        metadata_filters={}, expanded_queries=["What is AI?"],
    )


@pytest.mark.asyncio
async def test_planner_simple_query():
    from app.planner.planner import QueryPlanner
    planner = QueryPlanner()
    plan = await planner.plan("What is machine learning?")
    assert plan.query_type in list(QueryType)
    assert plan.retrieval_strategy in list(RetrievalStrategy)
    assert plan.expanded_queries


@pytest.mark.asyncio
async def test_planner_compliance_query():
    from app.planner.planner import QueryPlanner
    planner = QueryPlanner()
    plan = await planner.plan("What are the GDPR compliance requirements for data retention?")
    assert plan.query_type == QueryType.COMPLIANCE
    assert plan.require_verification is True
    assert plan.verification_level == "strict"


@pytest.mark.asyncio
async def test_planner_research_query():
    from app.planner.planner import QueryPlanner
    planner = QueryPlanner()
    plan = await planner.plan("Provide an overview of the history of artificial intelligence research")
    assert plan.query_type == QueryType.RESEARCH


@pytest.mark.asyncio
async def test_planner_kg_query():
    from app.planner.planner import QueryPlanner
    planner = QueryPlanner()
    plan = await planner.plan("Who is connected to the subsidiary acquired by the parent company?")
    assert plan.query_type == QueryType.KNOWLEDGE_GRAPH
    assert plan.use_graph is True


@pytest.mark.asyncio
async def test_rrf_fusion_merges_correctly():
    from app.hybrid.rrf import reciprocal_rank_fusion
    list_a = [
        RetrievedChunk(chunk_id="a", document_id="d", text="t", score=0.9, source="s", retrieval_method="bm25"),
        RetrievedChunk(chunk_id="b", document_id="d", text="t", score=0.7, source="s", retrieval_method="bm25"),
    ]
    list_b = [
        RetrievedChunk(chunk_id="b", document_id="d", text="t", score=0.8, source="s", retrieval_method="vector"),
        RetrievedChunk(chunk_id="c", document_id="d", text="t", score=0.6, source="s", retrieval_method="vector"),
    ]
    fused, scores = reciprocal_rank_fusion(list_a, list_b)
    assert len(fused) == 3
    assert all(c.retrieval_method == "fused" for c in fused)
    # 'b' appears in both lists → should rank highest
    assert fused[0].chunk_id == "b"


def test_metadata_filter_exact():
    from app.hybrid.metadata_filter import MetadataFilter
    from app.models.schemas import RetrievedChunk
    chunks = [
        RetrievedChunk(chunk_id="1", document_id="d", text="t", score=0.9,
                       source="s", metadata={"author": "Alice", "year": 2022}),
        RetrievedChunk(chunk_id="2", document_id="d", text="t", score=0.8,
                       source="s", metadata={"author": "Bob", "year": 2023}),
    ]
    result = MetadataFilter.apply(chunks, {"author": "Alice"})
    assert len(result) == 1
    assert result[0].chunk_id == "1"


def test_metadata_filter_range():
    from app.hybrid.metadata_filter import MetadataFilter
    from app.models.schemas import RetrievedChunk
    chunks = [
        RetrievedChunk(chunk_id="1", document_id="d", text="t", score=0.9,
                       source="s", metadata={"year": 2020}),
        RetrievedChunk(chunk_id="2", document_id="d", text="t", score=0.8,
                       source="s", metadata={"year": 2023}),
        RetrievedChunk(chunk_id="3", document_id="d", text="t", score=0.7,
                       source="s", metadata={"year": 2024}),
    ]
    result = MetadataFilter.apply(chunks, {"year": {"gte": 2022, "lte": 2023}})
    assert len(result) == 1
    assert result[0].chunk_id == "2"


def test_score_normalization():
    from app.hybrid.rrf import normalize_scores
    chunks = [
        RetrievedChunk(chunk_id="1", document_id="d", text="t", score=10.0, source="s"),
        RetrievedChunk(chunk_id="2", document_id="d", text="t", score=5.0, source="s"),
        RetrievedChunk(chunk_id="3", document_id="d", text="t", score=0.0, source="s"),
    ]
    normalized = normalize_scores(chunks)
    assert normalized[0].score == pytest.approx(1.0)
    assert normalized[2].score == pytest.approx(0.0)
    assert normalized[1].score == pytest.approx(0.5)

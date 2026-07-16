"""
app/core/dependencies.py
FastAPI dependency injection – constructs service graph once per process.
"""
from __future__ import annotations
from functools import lru_cache

from app.services.knowledge_graph.extractor import EntityRelationExtractor
from app.services.knowledge_graph.graph_service import KnowledgeGraphService
from app.services.knowledge_graph.neo4j_store import Neo4jStore
from app.services.planner.query_planner import QueryPlanner
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.query_expander import QueryExpander
from app.services.verification.verifier import AnswerVerifier
from app.services.pipeline import AgenticRAGPipeline


@lru_cache
def get_bm25() -> BM25Retriever:
    return BM25Retriever()


@lru_cache
def get_expander() -> QueryExpander:
    return QueryExpander()


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(bm25=get_bm25(), expander=get_expander())


@lru_cache
def get_neo4j_store() -> Neo4jStore:
    return Neo4jStore()


@lru_cache
def get_kg_service() -> KnowledgeGraphService:
    return KnowledgeGraphService(
        store=get_neo4j_store(),
        extractor=EntityRelationExtractor(),
    )


@lru_cache
def get_planner() -> QueryPlanner:
    return QueryPlanner()


@lru_cache
def get_verifier() -> AnswerVerifier:
    return AnswerVerifier()


@lru_cache
def get_security():
    """
    Build and initialise the SecurityPipeline singleton.
    Presidio engine is built once; reused across all requests.
    """
    from app.core.config import get_settings
    from app.security import SecurityPipeline
    cfg = get_settings()
    pipeline = SecurityPipeline(
        injection_threshold=cfg.security_injection_threshold,
        block_on_pii=cfg.security_block_on_pii,
        min_pii_score=cfg.security_pii_score_threshold,
    )
    pipeline.initialise()
    return pipeline


@lru_cache
def get_pipeline() -> AgenticRAGPipeline:
    return AgenticRAGPipeline(
        planner=get_planner(),
        retriever=get_hybrid_retriever(),
        kg=get_kg_service(),
        verifier=get_verifier(),
        security=get_security(),
    )

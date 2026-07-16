"""
Shared domain schemas — single source of truth for all phases.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    COMPLIANCE = "compliance"
    RESEARCH = "research"


class RetrievalStrategy(str, Enum):
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    GRAPH_ONLY = "graph_only"
    HYBRID_GRAPH = "hybrid_graph"


class QueryPlan(BaseModel):
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy
    use_graph: bool
    require_verification: bool
    verification_level: str = "standard"
    max_hops: int = 1
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    expanded_queries: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_method: str = "vector"


class HybridRetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    bm25_scores: dict[str, float] = Field(default_factory=dict)
    vector_scores: dict[str, float] = Field(default_factory=dict)
    fused_scores: dict[str, float] = Field(default_factory=dict)
    query_variants: list[str] = Field(default_factory=list)


class Entity(BaseModel):
    entity_id: str
    label: str
    text: str
    source_chunk_id: str
    confidence: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    rel_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    confidence: float = 1.0
    source_chunk_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphContext(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    expanded_chunk_ids: list[str] = Field(default_factory=list)
    traversal_depth: int = 0


class CitationValidation(BaseModel):
    citation_id: str
    is_valid: bool
    source_chunk_id: str
    claim_text: str
    supporting_text: str
    confidence: float


class ContradictionPair(BaseModel):
    chunk_id_a: str
    chunk_id_b: str
    claim_a: str
    claim_b: str
    contradiction_score: float


class VerificationResult(BaseModel):
    overall_confidence: float
    is_grounded: bool
    citations: list[CitationValidation] = Field(default_factory=list)
    contradictions: list[ContradictionPair] = Field(default_factory=list)
    hallucination_risk: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class RAGRequest(BaseModel):
    query: str
    metadata_filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    force_query_type: QueryType | None = None


class SecurityContext(BaseModel):
    """Security metadata attached to every RAGResponse."""
    session_id: str
    audit_id: str
    security_score: float
    injection_risk: float
    pii_entity_count: int
    pii_types: list[str] = Field(default_factory=list)
    masked: bool = False
    blocked: bool = False
    block_reason: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    answer: str
    query_plan: QueryPlan
    retrieved_chunks: list[RetrievedChunk]
    graph_context: GraphContext | None = None
    verification: VerificationResult | None = None
    security: SecurityContext | None = None          # ← new field
    latency_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None

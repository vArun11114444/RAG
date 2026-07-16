"""
app/services/pipeline.py
Main agentic pipeline orchestrator.
Wires Security → Planner → Hybrid Retrieval → Knowledge Graph → LLM → Verification.
"""
from __future__ import annotations
import time
import uuid

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.schemas import (
    AnswerResponse,
    GraphContext,
    QueryRequest,
    RetrievalRequest,
    VerificationResult,
)
from app.security import SecurityPipeline, SecurityRequest
from app.services.knowledge_graph.graph_service import KnowledgeGraphService
from app.services.observability.logger import get_logger
from app.services.observability.metrics import ACTIVE_QUERIES, QUERY_LATENCY, QUERY_TOTAL
from app.services.observability.tracing import trace_span
from app.services.planner.query_planner import QueryPlanner
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.verification.verifier import AnswerVerifier

_settings = get_settings()
_log = get_logger(__name__)

_ANSWER_SYSTEM = """You are a precise enterprise assistant.
Answer the question using ONLY the provided context passages.
Cite sources using bracket notation [chunk_id] after each claim.
If the context does not contain enough information, say so explicitly.
Never fabricate information."""


class AgenticRAGPipeline:

    def __init__(
        self,
        planner:   QueryPlanner,
        retriever: HybridRetriever,
        kg:        KnowledgeGraphService,
        verifier:  AnswerVerifier,
        security:  SecurityPipeline | None = None,
    ) -> None:
        self._planner   = planner
        self._retriever = retriever
        self._kg        = kg
        self._verifier  = verifier
        self._security  = security
        self._llm       = AsyncOpenAI(api_key=_settings.openai_api_key, base_url=_settings.openai_base_url)

    async def run(self, request: QueryRequest) -> AnswerResponse:
        trace_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        ACTIVE_QUERIES.inc()

        async with trace_span("pipeline", trace_id, {"query": request.query}):
            try:
                effective_query = request.query

                # ── Security Layer ────────────────────────────────────────────
                if self._security is not None:
                    sec_result = await self._security.run(
                        SecurityRequest(query=request.query, session_id=trace_id)
                    )
                    if sec_result.blocked:
                        _log.warning(
                            "request_blocked",
                            trace_id=trace_id,
                            reason=sec_result.block_reason,
                        )
                        from fastapi import HTTPException
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "security_violation",
                                "message": sec_result.block_reason,
                                "audit_id": sec_result.audit_id,
                            },
                        )
                    effective_query = sec_result.sanitized_query

                # ── Phase 4: Plan ─────────────────────────────────────────────
                decision = await self._planner.plan(effective_query, trace_id=trace_id)

                # ── Phase 1: Hybrid Retrieval ─────────────────────────────────
                retrieval_result = await self._retriever.retrieve(
                    RetrievalRequest(
                        query=effective_query,
                        top_k=_settings.final_top_k,
                        filters=request.filters,
                        expand_query=decision.use_hybrid_retrieval,
                    ),
                    trace_id=trace_id,
                )
                chunks = retrieval_result.chunks

                # ── Phase 2: Knowledge Graph ──────────────────────────────────
                graph_ctx: GraphContext | None = None
                if decision.use_knowledge_graph:
                    graph_ctx = await self._kg.build_context(
                        chunks,
                        max_hops=decision.max_hops,
                        trace_id=trace_id,
                    )

                # ── Build context window ──────────────────────────────────────
                context_parts: list[str] = [
                    f"[{c.chunk_id}] {c.content}" for c in chunks
                ]
                if graph_ctx and graph_ctx.expanded_passages:
                    context_parts.append(
                        "KNOWLEDGE GRAPH CONTEXT:\n" +
                        "\n".join(graph_ctx.expanded_passages)
                    )
                context_text = "\n\n".join(context_parts)

                # ── LLM generation ────────────────────────────────────────────
                resp = await self._llm.chat.completions.create(
                    model=_settings.llm_model,
                    messages=[
                        {"role": "system", "content": _ANSWER_SYSTEM},
                        {"role": "user", "content": (
                            f"CONTEXT:\n{context_text}\n\nQUESTION: {effective_query}"
                        )},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = resp.choices[0].message.content or ""

                # ── Phase 3: Verification ─────────────────────────────────────
                verification = await self._verifier.verify(
                    answer, chunks, trace_id=trace_id
                )

                if not verification.passed:
                    answer = (
                        f"⚠️ Low confidence answer (score: {verification.confidence_score:.2f}). "
                        f"Please verify independently.\n\n{answer}"
                    )

                latency_ms = (time.perf_counter() - t0) * 1000
                QUERY_LATENCY.observe(latency_ms / 1000)
                QUERY_TOTAL.labels(query_type=decision.query_type.value).inc()

                _log.info(
                    "pipeline_complete",
                    trace_id=trace_id,
                    query_type=decision.query_type,
                    confidence=verification.confidence_score,
                    latency_ms=round(latency_ms, 2),
                )

                return AnswerResponse(
                    query=request.query,
                    answer=answer,
                    chunks=chunks,
                    graph_context=graph_ctx,
                    verification=verification,
                    planner_decision=decision,
                    trace_id=trace_id,
                    latency_ms=round(latency_ms, 2),
                )
            finally:
                ACTIVE_QUERIES.dec()

_settings = get_settings()
_log = get_logger(__name__)

_ANSWER_SYSTEM = """You are a precise enterprise assistant.
Answer the question using ONLY the provided context passages.
Cite sources using bracket notation [chunk_id] after each claim.
If the context does not contain enough information, say so explicitly.
Never fabricate information."""


class AgenticRAGPipeline:

    def __init__(
        self,
        planner:   QueryPlanner,
        retriever: HybridRetriever,
        kg:        KnowledgeGraphService,
        verifier:  AnswerVerifier,
    ) -> None:
        self._planner   = planner
        self._retriever = retriever
        self._kg        = kg
        self._verifier  = verifier
        self._llm       = AsyncOpenAI(api_key=_settings.openai_api_key, base_url=_settings.openai_base_url)

    async def run(self, request: QueryRequest) -> AnswerResponse:
        trace_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        ACTIVE_QUERIES.inc()

        async with trace_span("pipeline", trace_id, {"query": request.query}):
            try:
                # ── Phase 4: Plan ─────────────────────────────────────────────
                decision = await self._planner.plan(request.query, trace_id=trace_id)

                # ── Phase 1: Hybrid Retrieval ─────────────────────────────────
                retrieval_result = await self._retriever.retrieve(
                    RetrievalRequest(
                        query=request.query,
                        top_k=_settings.final_top_k,
                        filters=request.filters,
                        expand_query=decision.use_hybrid_retrieval,
                    ),
                    trace_id=trace_id,
                )
                chunks = retrieval_result.chunks

                # ── Phase 2: Knowledge Graph ──────────────────────────────────
                graph_ctx: GraphContext | None = None
                if decision.use_knowledge_graph:
                    graph_ctx = await self._kg.build_context(
                        chunks,
                        max_hops=decision.max_hops,
                        trace_id=trace_id,
                    )

                # ── Build context window ──────────────────────────────────────
                context_parts: list[str] = [
                    f"[{c.chunk_id}] {c.content}" for c in chunks
                ]
                if graph_ctx and graph_ctx.expanded_passages:
                    context_parts.append(
                        "KNOWLEDGE GRAPH CONTEXT:\n" +
                        "\n".join(graph_ctx.expanded_passages)
                    )
                context_text = "\n\n".join(context_parts)

                # ── LLM generation ────────────────────────────────────────────
                resp = await self._llm.chat.completions.create(
                    model=_settings.llm_model,
                    messages=[
                        {"role": "system", "content": _ANSWER_SYSTEM},
                        {"role": "user", "content": (
                            f"CONTEXT:\n{context_text}\n\nQUESTION: {request.query}"
                        )},
                    ],
                    temperature=0.1,
                    max_tokens=1024,
                )
                answer = resp.choices[0].message.content or ""

                # ── Phase 3: Verification ─────────────────────────────────────
                verification = await self._verifier.verify(
                    answer, chunks, trace_id=trace_id
                )

                # Flag low-confidence answers
                if not verification.passed:
                    answer = (
                        f"⚠️ Low confidence answer (score: {verification.confidence_score:.2f}). "
                        f"Please verify independently.\n\n{answer}"
                    )

                latency_ms = (time.perf_counter() - t0) * 1000
                QUERY_LATENCY.observe(latency_ms / 1000)
                QUERY_TOTAL.labels(query_type=decision.query_type.value).inc()

                _log.info(
                    "pipeline_complete",
                    trace_id=trace_id,
                    query_type=decision.query_type,
                    confidence=verification.confidence_score,
                    latency_ms=round(latency_ms, 2),
                )

                return AnswerResponse(
                    query=request.query,
                    answer=answer,
                    chunks=chunks,
                    graph_context=graph_ctx,
                    verification=verification,
                    planner_decision=decision,
                    trace_id=trace_id,
                    latency_ms=round(latency_ms, 2),
                )
            finally:
                ACTIVE_QUERIES.dec()

"""
Pipeline executor — ties all five phases together with the Security Layer.

Request flow:
    Security → Plan → Retrieve → Graph → Generate → Verify → Return

The security layer runs FIRST — blocked requests never reach the planner.
Sanitized (PII-masked) queries are used for all downstream phases.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.graph.pipeline import GraphPipeline
from app.hybrid.retriever import HybridRetriever
from app.models.schemas import (
    GraphContext,
    QueryType,
    RAGRequest,
    RAGResponse,
    RetrievedChunk,
    SecurityContext,
    VerificationResult,
)
from app.observability import (
    ACTIVE_REQUESTS,
    ERRORS,
    PHASE_LATENCY,
    REQUEST_LATENCY,
    REQUEST_TOTAL,
    Timer,
    clear_context,
    get_logger,
    set_context,
)
from app.observability.tracer import trace_span
from app.planner.planner import QueryPlanner
from app.security import SecurityPipeline, SecurityRequest, SecurityResult
from app.verification.pipeline import VerificationPipeline

log = get_logger(__name__)
settings = get_settings()

_ANSWER_PROMPT = """\
You are a precise, source-grounded assistant.
Answer the question using ONLY the provided context chunks.
If the answer is not in the context, say "I don't have enough information."
Cite sources as [SOURCE: filename, page N].

Question: {query}

Context:
{context}

Answer:"""


class RAGPipelineExecutor:
    """
    Orchestrates the full enterprise RAG pipeline:
        Security → Plan → Retrieve → Graph → Generate → Verify → Return
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        graph_pipeline: GraphPipeline,
        verifier: VerificationPipeline,
        planner: QueryPlanner,
        security: SecurityPipeline | None = None,
    ) -> None:
        self._retriever = hybrid_retriever
        self._graph = graph_pipeline
        self._verifier = verifier
        self._planner = planner
        self._security = security
        self._oai = (
            AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
            if settings.OPENAI_API_KEY
            else None
        )

    async def run(self, request: RAGRequest) -> RAGResponse:
        trace_id = str(uuid.uuid4())
        set_context(trace_id=trace_id, query=request.query[:80])
        ACTIVE_REQUESTS.inc()
        t_total = time.perf_counter()
        latency: dict[str, float] = {}

        try:
            async with trace_span("full_pipeline", {"query": request.query}, trace_id):

                # ── Security Layer ─────────────────────────────────────────
                security_result: SecurityResult | None = None
                effective_query = request.query

                if self._security is not None and settings.SECURITY_ENABLED:
                    with Timer("security", latency):
                        sec_req = SecurityRequest(
                            query=request.query,
                            session_id=trace_id,
                        )
                        security_result = await self._security.run(sec_req)
                    PHASE_LATENCY.labels(phase="security").observe(
                        latency["security"] / 1000
                    )

                    if security_result.blocked:
                        log.warning(
                            "pipeline_request_blocked",
                            extra={
                                "trace_id": trace_id,
                                "reason": security_result.block_reason,
                                "event_type": (
                                    security_result.block_event_type.value
                                    if security_result.block_event_type else "unknown"
                                ),
                            },
                        )
                        REQUEST_TOTAL.labels(
                            query_type="blocked", status="blocked"
                        ).inc()
                        # Return a structured blocked response instead of raising
                        # (callers that need HTTP 400 should check response.security.blocked)
                        return _blocked_response(
                            request=request,
                            security_result=security_result,
                            trace_id=trace_id,
                            latency=latency,
                        )

                    # Use PII-masked query for all downstream phases
                    effective_query = security_result.sanitized_query

                # ── Phase 4: Plan ─────────────────────────────────────────
                with Timer("plan", latency):
                    plan = await self._planner.plan(
                        effective_query,
                        metadata_filters=request.metadata_filters,
                        force_type=request.force_query_type,
                    )
                PHASE_LATENCY.labels(phase="plan").observe(latency["plan"] / 1000)

                # ── Phase 1: Hybrid Retrieval ─────────────────────────────
                with Timer("retrieval", latency):
                    retrieval_result = await self._retriever.retrieve(
                        query=effective_query,
                        top_k=request.top_k,
                        metadata_filters=plan.metadata_filters or None,
                        expand=True,
                    )
                    plan.expanded_queries = retrieval_result.query_variants
                chunks = retrieval_result.chunks
                PHASE_LATENCY.labels(phase="retrieval").observe(
                    latency["retrieval"] / 1000
                )

                # ── Phase 2: Knowledge Graph ──────────────────────────────
                graph_context: GraphContext | None = None
                if plan.use_graph:
                    with Timer("graph", latency):
                        graph_context = await self._graph.enrich(
                            chunks,
                            max_hops=plan.max_hops,
                            store_to_graph=False,
                        )
                    PHASE_LATENCY.labels(phase="graph").observe(
                        latency["graph"] / 1000
                    )

                # ── Generate answer ───────────────────────────────────────
                with Timer("generation", latency):
                    answer = await self._generate_answer(effective_query, chunks)
                PHASE_LATENCY.labels(phase="generation").observe(
                    latency["generation"] / 1000
                )

                # ── Phase 3: Verification ─────────────────────────────────
                verification: VerificationResult | None = None
                if plan.require_verification:
                    with Timer("verification", latency):
                        verification = await self._verifier.verify(
                            answer=answer,
                            chunks=chunks,
                            level=plan.verification_level,
                        )
                    PHASE_LATENCY.labels(phase="verification").observe(
                        latency["verification"] / 1000
                    )

                    # Compliance: prepend low-confidence warning
                    if (
                        plan.query_type == QueryType.COMPLIANCE
                        and verification
                        and verification.overall_confidence < settings.CONFIDENCE_THRESHOLD
                    ):
                        answer = (
                            f"⚠️ Low confidence ({verification.overall_confidence:.0%}) "
                            f"— please verify with authoritative sources.\n\n{answer}"
                        )

                total_ms = round((time.perf_counter() - t_total) * 1000, 2)
                latency["total"] = total_ms
                REQUEST_TOTAL.labels(query_type=plan.query_type, status="ok").inc()
                REQUEST_LATENCY.labels(query_type=plan.query_type).observe(
                    total_ms / 1000
                )

                log.info(
                    "pipeline_complete",
                    extra={
                        "trace_id": trace_id,
                        "query_type": plan.query_type,
                        "chunks": len(chunks),
                        "latency_ms": latency,
                        "confidence": (
                            verification.overall_confidence if verification else None
                        ),
                        "security_score": (
                            security_result.security_score if security_result else None
                        ),
                    },
                )

                return RAGResponse(
                    answer=answer,
                    query_plan=plan,
                    retrieved_chunks=chunks,
                    graph_context=graph_context,
                    verification=verification,
                    security=_build_security_context(security_result),
                    latency_ms=latency,
                    trace_id=trace_id,
                )

        except Exception as exc:
            ERRORS.labels(phase="pipeline", error_type=type(exc).__name__).inc()
            REQUEST_TOTAL.labels(
                query_type=request.force_query_type or "unknown", status="error"
            ).inc()
            log.error(
                "pipeline_failed",
                extra={"error": str(exc), "trace_id": trace_id},
                exc_info=True,
            )
            raise
        finally:
            ACTIVE_REQUESTS.dec()
            clear_context()

    async def _generate_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Generate an answer from retrieved chunks using the LLM."""
        if not chunks:
            return "No relevant documents found to answer this query."

        context = "\n\n---\n\n".join(
            f"[SOURCE: {c.source}, page {c.page or 'N/A'}]\n{c.text}"
            for c in chunks[:8]
        )
        prompt = _ANSWER_PROMPT.format(query=query, context=context)

        if self._oai is None:
            return f"Retrieved {len(chunks)} relevant passages:\n\n{context[:2000]}"

        try:
            resp = await self._oai.chat.completions.create(
                model=settings.OPENAI_MODEL,
                temperature=settings.OPENAI_TEMPERATURE,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or "No answer generated."
        except Exception as exc:
            ERRORS.labels(phase="generation", error_type=type(exc).__name__).inc()
            log.error("generation_failed", extra={"error": str(exc)})
            return f"Answer generation failed: {exc}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_security_context(
    security_result: SecurityResult | None,
) -> SecurityContext | None:
    """Convert SecurityResult → SecurityContext for the RAGResponse."""
    if security_result is None:
        return None
    return SecurityContext(
        session_id=security_result.session_id,
        audit_id=security_result.audit_id,
        security_score=security_result.security_score,
        injection_risk=security_result.injection_risk,
        pii_entity_count=security_result.pii_entity_count,
        pii_types=list({
            e.entity_type
            for e in (
                security_result.pii_result.entities
                if security_result.pii_result else []
            )
        }),
        masked=(
            security_result.masking_result is not None
            and security_result.masking_result.has_masked_content
        ),
        blocked=security_result.blocked,
        block_reason=security_result.block_reason,
        latency_ms=security_result.latency_ms,
    )


def _blocked_response(
    request: RAGRequest,
    security_result: SecurityResult,
    trace_id: str,
    latency: dict[str, float],
) -> RAGResponse:
    """Build a structured RAGResponse for blocked requests."""
    from app.models.schemas import QueryPlan, QueryType, RetrievalStrategy

    dummy_plan = QueryPlan(
        query_type=QueryType.SIMPLE,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        use_graph=False,
        require_verification=False,
        expanded_queries=[request.query],
    )
    return RAGResponse(
        answer=(
            f"Request blocked by security layer: {security_result.block_reason}"
        ),
        query_plan=dummy_plan,
        retrieved_chunks=[],
        security=_build_security_context(security_result),
        latency_ms=latency,
        trace_id=trace_id,
    )

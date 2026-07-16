"""
app/api/routes/query.py
Main query endpoint + BM25 admin endpoint + health check.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.config import get_settings
from app.models.schemas import (
    RAGRequest, RAGResponse, QueryPlan, QueryType, RetrievalStrategy,
    HybridRetrievalResult, GraphContext, VerificationResult, RetrievedChunk,
    SecurityContext,
)
from app.observability.logger import get_logger, set_context, clear_context
from app.observability.metrics import (
    REQUEST_TOTAL, REQUEST_LATENCY, ACTIVE_REQUESTS, ERRORS,
    CONFIDENCE_SCORE, HALLUCINATION_RISK,
)
from app.observability.tracer import trace_span

router = APIRouter(prefix="/api/v2", tags=["Agentic RAG"])
log = get_logger(__name__)
settings = get_settings()


def _build_dependencies(request: Request) -> dict[str, Any]:
    """Lazily construct pipeline components; stored on app.state after first call."""
    state = request.app.state

    if not hasattr(state, "bm25"):
        from app.hybrid.bm25_retriever import BM25Retriever
        state.bm25 = BM25Retriever()

    if not hasattr(state, "hybrid_retriever"):
        from app.hybrid.retriever import HybridRetriever
        state.hybrid_retriever = HybridRetriever(bm25=state.bm25)

    if not hasattr(state, "graph_pipeline"):
        from app.graph.pipeline import GraphPipeline
        state.graph_pipeline = GraphPipeline(neo4j=getattr(state, "neo4j", None))

    if not hasattr(state, "verification_pipeline"):
        from app.verification.pipeline import VerificationPipeline
        state.verification_pipeline = VerificationPipeline()

    if not hasattr(state, "planner"):
        from app.planner.planner import QueryPlanner
        state.planner = QueryPlanner()

    # Security pipeline — already initialised in lifespan; fall back if absent
    if not hasattr(state, "security"):
        from app.security import SecurityPipeline
        state.security = SecurityPipeline()
        state.security.initialise()

    return {
        "bm25": state.bm25,
        "hybrid": state.hybrid_retriever,
        "graph": state.graph_pipeline,
        "verify": state.verification_pipeline,
        "planner": state.planner,
        "security": state.security,
    }


@router.post("/query", response_model=RAGResponse)
async def query(body: RAGRequest, request: Request) -> RAGResponse:
    """
    Main agentic RAG endpoint.

    Pipeline:
        1. Planner classifies query → QueryPlan
        2. Hybrid retriever (BM25 + vector + RRF)
        3. Optional Knowledge Graph context expansion
        4. LLM answer generation
        5. Verification layer (grounding + citations + hallucination)
    """
    trace_id = str(uuid.uuid4())
    set_context(trace_id=trace_id, query=body.query[:80])
    latency: dict[str, float] = {}
    ACTIVE_REQUESTS.inc()

    try:
        async with trace_span("full_pipeline", inputs={"query": body.query}, run_id=trace_id):
            deps = _build_dependencies(request)
            t0 = time.perf_counter()

            # ── Security Layer (runs before everything else) ───────────────────
            from app.security import SecurityRequest, SecurityResult
            from app.security.metrics import SECURITY_BLOCKS_TOTAL

            security_result: SecurityResult | None = None
            effective_query = body.query

            if deps["security"] is not None:
                async with trace_span("security"):
                    t_sec = time.perf_counter()
                    sec_req = SecurityRequest(
                        query=body.query,
                        session_id=trace_id,
                    )
                    security_result = await deps["security"].run(sec_req)
                    latency["security_ms"] = round((time.perf_counter() - t_sec) * 1000, 2)

                if security_result.blocked:
                    log.warning(
                        "request_blocked_by_security",
                        extra={
                            "trace_id": trace_id,
                            "reason": security_result.block_reason,
                            "event_type": security_result.block_event_type.value
                                if security_result.block_event_type else "unknown",
                        },
                    )
                    REQUEST_TOTAL.labels(query_type="blocked", status="blocked").inc()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "security_violation",
                            "message": security_result.block_reason,
                            "event_type": security_result.block_event_type.value
                                if security_result.block_event_type else "unknown",
                            "audit_id": security_result.audit_id,
                        },
                    )

                # Use sanitized (PII-masked) query for all downstream phases
                effective_query = security_result.sanitized_query

            # ── Phase 4: Plan ─────────────────────────────────────────────────
            async with trace_span("planner"):
                t = time.perf_counter()
                plan: QueryPlan = await deps["planner"].plan(
                    query=effective_query,
                    metadata_filters=body.metadata_filters,
                    force_type=body.force_query_type,
                )
                latency["planner_ms"] = round((time.perf_counter() - t) * 1000, 2)

            log.info(
                "query_planned",
                extra={
                    "query_type": plan.query_type,
                    "strategy": plan.retrieval_strategy,
                    "use_graph": plan.use_graph,
                },
            )

            # ── Phase 1: Hybrid Retrieval ─────────────────────────────────────
            async with trace_span("hybrid_retrieval"):
                t = time.perf_counter()
                hybrid_result: HybridRetrievalResult = await deps["hybrid"].retrieve(
                    query=effective_query,
                    top_k=body.top_k,
                    metadata_filters={**body.metadata_filters, **plan.metadata_filters},
                    expanded_queries=plan.expanded_queries,
                    strategy=plan.retrieval_strategy,
                )
                latency["hybrid_ms"] = round((time.perf_counter() - t) * 1000, 2)

            # ── Phase 2: Knowledge Graph (conditional) ────────────────────────
            graph_ctx: GraphContext | None = None
            if plan.use_graph and deps["graph"].is_available():
                async with trace_span("knowledge_graph"):
                    t = time.perf_counter()
                    graph_ctx = await deps["graph"].run(
                        query=effective_query,
                        chunks=hybrid_result.chunks,
                        max_hops=plan.max_hops,
                    )
                    latency["graph_ms"] = round((time.perf_counter() - t) * 1000, 2)

            # ── LLM Answer Generation ─────────────────────────────────────────
            async with trace_span("llm_generation"):
                t = time.perf_counter()
                answer = await _generate_answer(
                    query=effective_query,
                    chunks=hybrid_result.chunks,
                    graph_ctx=graph_ctx,
                    plan=plan,
                )
                latency["llm_ms"] = round((time.perf_counter() - t) * 1000, 2)

            # ── Phase 3: Verification (conditional) ───────────────────────────
            verification: VerificationResult | None = None
            if plan.require_verification:
                async with trace_span("verification"):
                    t = time.perf_counter()
                    verification = await deps["verify"].run(
                        answer=answer,
                        chunks=hybrid_result.chunks,
                        level=plan.verification_level,
                    )
                    latency["verify_ms"] = round((time.perf_counter() - t) * 1000, 2)

                    CONFIDENCE_SCORE.observe(verification.overall_confidence)
                    HALLUCINATION_RISK.observe(verification.hallucination_risk)

            latency["total_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            REQUEST_TOTAL.labels(query_type=plan.query_type, status="ok").inc()
            REQUEST_LATENCY.labels(query_type=plan.query_type).observe(
                latency["total_ms"] / 1000
            )

            # Build SecurityContext for the response
            sec_ctx: SecurityContext | None = None
            if security_result is not None:
                sec_ctx = SecurityContext(
                    session_id=security_result.session_id,
                    audit_id=security_result.audit_id,
                    security_score=security_result.security_score,
                    injection_risk=security_result.injection_risk,
                    pii_entity_count=security_result.pii_entity_count,
                    pii_types=list({
                        e.entity_type
                        for e in (security_result.pii_result.entities
                                  if security_result.pii_result else [])
                    }),
                    masked=security_result.masking_result is not None
                           and security_result.masking_result.has_masked_content,
                    blocked=False,
                    latency_ms=security_result.latency_ms,
                )

            return RAGResponse(
                answer=answer,
                query_plan=plan,
                retrieved_chunks=hybrid_result.chunks,
                graph_context=graph_ctx,
                verification=verification,
                security=sec_ctx,
                latency_ms=latency,
                trace_id=trace_id,
            )

    except HTTPException:
        raise   # Security blocks and validation errors — pass through as-is
    except Exception as exc:
        REQUEST_TOTAL.labels(query_type="unknown", status="error").inc()
        ERRORS.labels(phase="api", error_type=type(exc).__name__).inc()
        log.error("query_failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )
    finally:
        ACTIVE_REQUESTS.dec()
        clear_context()


@router.post("/admin/index-bm25", status_code=status.HTTP_202_ACCEPTED)
async def index_bm25(chunks: list[dict], request: Request) -> dict:
    """
    Re-index BM25 from a list of chunk dicts.
    Call this after ingesting new documents via the existing PDF pipeline.
    Expected chunk dict keys: chunk_id, document_id, text, source, page, metadata.
    """
    deps = _build_dependencies(request)
    await deps["bm25"].build_index(chunks)
    return {"status": "indexed", "count": len(chunks)}


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_files(
    request: Request,
    files: list[Any] = None,  # FastAPI UploadFile injected via Form
) -> dict:
    """
    Validate and stage uploaded files through the security layer.
    Files that pass all checks are returned with metadata for downstream ingestion.

    Requires python-multipart to be installed.
    Use multipart/form-data with field name 'files'.
    """
    from fastapi import UploadFile, Form
    from fastapi.datastructures import UploadFile as UploadFileType
    from app.security.file_validator import UploadedFile as SecUploadedFile

    # Re-parse the real multipart files from the request
    form = await request.form()
    raw_files = form.getlist("files")

    if not raw_files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No files provided. Send files as multipart/form-data with field name 'files'.",
        )

    deps = _build_dependencies(request)
    security = deps.get("security")
    results = []

    for raw_file in raw_files:
        filename = getattr(raw_file, "filename", "unknown")
        content_type = getattr(raw_file, "content_type", "application/octet-stream")

        try:
            content: bytes = await raw_file.read()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read file '{filename}': {exc}",
            )

        if security is not None:
            from app.security import SecurityRequest
            from app.security.file_validator import UploadedFile as SecUploadedFile

            sec_file = SecUploadedFile(
                filename=filename,
                content_type=content_type,
                content=content,
            )
            sec_result = await security.run(
                SecurityRequest(
                    query="",
                    uploaded_files=[sec_file],
                    session_id=str(uuid.uuid4()),
                )
            )

            if sec_result.blocked:
                log.warning(
                    "upload_blocked",
                    extra={
                        "file_name": filename,
                        "reason": sec_result.block_reason,
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "file_rejected",
                        "file": filename,
                        "reason": sec_result.block_reason,
                        "audit_id": sec_result.audit_id,
                    },
                )

            file_meta = sec_result.file_results[0] if sec_result.file_results else None

            # Upload to Supabase Storage after security passes
            storage_url: str | None = None
            try:
                from app.storage.supabase_storage import SupabaseStorageService
                storage = SupabaseStorageService()
                storage_url = await storage.upload(
                    file_data=content,
                    filename=filename,
                    content_type=content_type,
                )
            except Exception as exc:
                log.warning(
                    "supabase_upload_skipped",
                    extra={"file_name": filename, "error": str(exc)},
                )

            results.append({
                "filename": filename,
                "size_bytes": len(content),
                "detected_mime": file_meta.detected_mime if file_meta else content_type,
                "warnings": file_meta.warnings if file_meta else [],
                "status": "accepted",
                "storage_url": storage_url,
                "audit_id": sec_result.audit_id,
            })
        else:
            # No security layer — still upload to Supabase
            storage_url = None
            try:
                from app.storage.supabase_storage import SupabaseStorageService
                storage = SupabaseStorageService()
                storage_url = await storage.upload(
                    file_data=content,
                    filename=filename,
                    content_type=content_type,
                )
            except Exception as exc:
                log.warning(
                    "supabase_upload_skipped",
                    extra={"file_name": filename, "error": str(exc)},
                )
            results.append({
                "filename": filename,
                "size_bytes": len(content),
                "status": "accepted",
                "storage_url": storage_url,
            })

    return {
        "files_processed": len(results),
        "results": results,
    }


@router.get("/health")
async def health(request: Request) -> dict:
    """Health check — includes security layer status."""
    state = request.app.state
    security_ready = (
        hasattr(state, "security") and state.security is not None
    )
    neo4j_connected = (
        hasattr(state, "neo4j") and state.neo4j is not None
    )
    return {
        "status": "ok",
        "version": "2.0.0",
        "components": {
            "security": "ready" if security_ready else "disabled",
            "neo4j": "connected" if neo4j_connected else "unavailable",
        },
    }


# ── Internal: LLM answer generation ──────────────────────────────────────────

async def _generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    graph_ctx: GraphContext | None,
    plan: QueryPlan,
) -> str:
    """
    Assemble context from chunks (+ optional graph) and call the LLM.
    Returns the generated answer string.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)

    # Build context block
    context_parts = []
    for i, chunk in enumerate(chunks[:8], 1):
        context_parts.append(f"[{i}] (source: {chunk.source}) {chunk.text}")

    if graph_ctx and graph_ctx.entities:
        entity_lines = [
            f"- {e.label}: {e.text}" for e in graph_ctx.entities[:10]
        ]
        rel_lines = [
            f"- {r.source_entity_id} --[{r.relation_type}]--> {r.target_entity_id}"
            for r in graph_ctx.relationships[:10]
        ]
        context_parts.append(
            "\n\nKnowledge Graph Context:\nEntities:\n"
            + "\n".join(entity_lines)
            + "\nRelationships:\n"
            + "\n".join(rel_lines)
        )

    context = "\n\n".join(context_parts)

    system = (
        "You are a precise, grounded research assistant. "
        "Answer using ONLY the provided context. "
        "Cite sources as [1], [2], etc. "
        "If the answer is not in the context, say so clearly."
    )
    if plan.query_type == QueryType.COMPLIANCE:
        system += (
            " This is a compliance query — be conservative, cite every claim, "
            "and flag any uncertainty explicitly."
        )

    resp = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        temperature=settings.OPENAI_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content or ""

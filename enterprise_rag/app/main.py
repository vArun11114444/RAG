"""
app/main.py — FastAPI application factory with lifespan management.
Connects to Neo4j, starts Prometheus, registers all routers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.observability.logger import configure_logging, get_logger
from app.observability.metrics import start_metrics_server
from app.observability.tracer import init_langsmith
from app.api.routes.query import router as query_router

_settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(_settings.LOG_LEVEL)
    log = get_logger("startup")

    # LangSmith tracing
    init_langsmith(
        api_key=_settings.LANGCHAIN_API_KEY,
        project=_settings.LANGCHAIN_PROJECT,
        endpoint=_settings.LANGCHAIN_ENDPOINT,
        enabled=_settings.LANGCHAIN_TRACING_V2,
    )

    # Security pipeline (initialise Presidio engine)
    if _settings.SECURITY_ENABLED:
        from app.security import SecurityPipeline
        security = SecurityPipeline(
            injection_threshold=_settings.SECURITY_INJECTION_THRESHOLD,
            block_on_pii=_settings.SECURITY_BLOCK_ON_PII,
            min_pii_score=_settings.SECURITY_PII_SCORE_THRESHOLD,
        )
        security.initialise()
        app.state.security = security
        log.info("security_pipeline_ready")
    else:
        app.state.security = None
        log.warning("security_pipeline_disabled")

    # Neo4j connection (non-fatal if unavailable in dev)
    from app.graph.neo4j_client import Neo4jClient
    neo4j = Neo4jClient()
    try:
        await neo4j.connect()
        log.info("neo4j_connected")
        app.state.neo4j = neo4j
    except Exception as exc:
        log.warning("neo4j_unavailable", extra={"error": str(exc)})
        app.state.neo4j = None

    # Prometheus metrics server
    if _settings.METRICS_ENABLED:
        try:
            start_metrics_server(_settings.METRICS_PORT)
            log.info("prometheus_started", extra={"port": _settings.METRICS_PORT})
        except Exception as exc:
            log.warning("prometheus_failed", extra={"error": str(exc)})

    log.info("app_ready", extra={"env": _settings.ENV})
    yield

    # Shutdown
    if app.state.neo4j:
        await app.state.neo4j.close()
    log.info("app_shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise Agentic RAG",
        description=(
            "5-phase agentic retrieval: "
            "Hybrid Retrieval → Knowledge Graph → Verification → Planner → Observability"
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(query_router)
    return app


app = create_app()

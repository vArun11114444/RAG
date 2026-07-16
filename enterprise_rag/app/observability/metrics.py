"""
Prometheus metrics — counters, histograms, gauges.
Imported once at startup; labels let us slice by phase and query_type.
"""
from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ── Request lifecycle ─────────────────────────────────────────────────────────
REQUEST_TOTAL = Counter(
    "rag_requests_total",
    "Total RAG pipeline requests",
    ["query_type", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "End-to-end request latency",
    ["query_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Per-phase latency ─────────────────────────────────────────────────────────
PHASE_LATENCY = Histogram(
    "rag_phase_latency_seconds",
    "Latency per pipeline phase",
    ["phase"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

# ── Retrieval quality ─────────────────────────────────────────────────────────
RETRIEVAL_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks returned per request",
    ["strategy"],
    buckets=[1, 3, 5, 10, 20, 50],
)

BM25_SCORE = Histogram(
    "rag_bm25_score",
    "BM25 top-1 score distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Verification ──────────────────────────────────────────────────────────────
CONFIDENCE_SCORE = Histogram(
    "rag_confidence_score",
    "Overall verification confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

HALLUCINATION_RISK = Histogram(
    "rag_hallucination_risk",
    "Hallucination risk per request",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CONTRADICTIONS_FOUND = Counter(
    "rag_contradictions_total",
    "Total contradictions detected across responses",
)

# ── Graph ─────────────────────────────────────────────────────────────────────
GRAPH_ENTITIES = Histogram(
    "rag_graph_entities",
    "Entities extracted per request",
    buckets=[0, 1, 5, 10, 20, 50, 100],
)

GRAPH_TRAVERSAL_DEPTH = Histogram(
    "rag_graph_traversal_depth",
    "Hops taken during graph traversal",
    buckets=[0, 1, 2, 3],
)

# ── Errors ────────────────────────────────────────────────────────────────────
ERRORS = Counter(
    "rag_errors_total",
    "Total errors by phase",
    ["phase", "error_type"],
)

# ── System ────────────────────────────────────────────────────────────────────
ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Requests currently being processed",
)


def start_metrics_server(port: int = 9090) -> None:
    start_http_server(port)

"""
app/services/observability/metrics.py
Prometheus metrics for every pipeline phase.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from app.core.config import get_settings

_settings = get_settings()

# ── Counters ──────────────────────────────────────────────────────────────────
QUERY_TOTAL = Counter(
    "rag_queries_total", "Total queries processed",
    ["query_type"],
)
VERIFICATION_PASS = Counter(
    "rag_verification_pass_total", "Verification checks passed",
    ["depth"],
)
VERIFICATION_FAIL = Counter(
    "rag_verification_fail_total", "Verification checks failed",
    ["reason"],
)

# ── Histograms (latency) ──────────────────────────────────────────────────────
QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds", "End-to-end query latency",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10],
)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds", "Hybrid retrieval latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2],
)
GRAPH_LATENCY = Histogram(
    "rag_graph_latency_seconds", "Knowledge graph traversal latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1],
)
VERIFY_LATENCY = Histogram(
    "rag_verify_latency_seconds", "Verification layer latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
ACTIVE_QUERIES = Gauge("rag_active_queries", "Currently in-flight queries")
RRF_TOP_SCORE  = Gauge("rag_rrf_top_score", "RRF score of the top-ranked chunk")
CONFIDENCE     = Gauge("rag_last_confidence_score", "Confidence score of last answer")


def start_metrics_server() -> None:
    start_http_server(_settings.prometheus_port)

"""
Security Prometheus metrics — app/security/metrics.py

Extends the existing observability metrics without modifying metrics.py.
All metric names are prefixed `rag_security_` to namespace cleanly.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Request lifecycle ─────────────────────────────────────────────────────────

SECURITY_CHECKS_TOTAL = Counter(
    "rag_security_checks_total",
    "Total number of security pipeline checks run",
)

SECURITY_BLOCKS_TOTAL = Counter(
    "rag_security_blocks_total",
    "Total requests blocked by the security layer",
    ["reason"],   # injection | pii | file_rejected | malicious | file_too_large
)

# ── PII ───────────────────────────────────────────────────────────────────────

SECURITY_PII_ENTITIES = Histogram(
    "rag_security_pii_entities",
    "Number of PII entities detected per request",
    buckets=[0, 1, 2, 3, 5, 10, 20],
)

SECURITY_PII_BY_TYPE = Counter(
    "rag_security_pii_by_type_total",
    "PII detections broken down by entity type",
    ["entity_type"],
)

# ── Injection ─────────────────────────────────────────────────────────────────

SECURITY_INJECTION_RISK = Histogram(
    "rag_security_injection_risk",
    "Distribution of injection risk scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

SECURITY_INJECTION_BY_CATEGORY = Counter(
    "rag_security_injection_by_category_total",
    "Injection detections by attack category",
    ["category"],
)

# ── File validation ───────────────────────────────────────────────────────────

SECURITY_FILE_REJECTIONS = Counter(
    "rag_security_file_rejections_total",
    "Files rejected during validation, by reason",
    ["reason"],   # malicious | invalid_mime | file_too_large | file_rejected
)

SECURITY_FILE_SIZE = Histogram(
    "rag_security_file_size_bytes",
    "Size distribution of accepted uploaded files",
    buckets=[
        1_024,           # 1 KB
        10_240,          # 10 KB
        102_400,         # 100 KB
        1_048_576,       # 1 MB
        10_485_760,      # 10 MB
        52_428_800,      # 50 MB
    ],
)

# ── Latency ───────────────────────────────────────────────────────────────────

SECURITY_LATENCY = Histogram(
    "rag_security_latency_seconds",
    "End-to-end security pipeline latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# ── Health ────────────────────────────────────────────────────────────────────

SECURITY_PRESIDIO_AVAILABLE = Gauge(
    "rag_security_presidio_available",
    "1 if Presidio engine is available, 0 if using regex fallback",
)

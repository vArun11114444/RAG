"""
tests/conftest.py

Shared pytest fixtures used across all test modules.
Stubs external packages so tests run without installed ML models or services.
"""
from __future__ import annotations

import io
import sys
import types
import uuid
import zipfile
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Stub external packages before any app import ──────────────────────────────

def _make_fake_metric():
    class FakeMetric:
        def __init__(self, *a, **k): pass
        def inc(self, amount=1): pass
        def observe(self, v): pass
        def labels(self, **k): return self
        def set(self, v): pass
    return FakeMetric()


def _stub_prometheus():
    """Stub prometheus_client so tests don't need it installed."""
    if "prometheus_client" in sys.modules:
        return
    prom = types.ModuleType("prometheus_client")

    class FakeMetricClass:
        def __init__(self, *a, **k): pass
        def inc(self, amount=1): pass
        def observe(self, v): pass
        def labels(self, **k): return self
        def set(self, v): pass

    prom.Counter = FakeMetricClass
    prom.Histogram = FakeMetricClass
    prom.Gauge = FakeMetricClass
    prom.start_http_server = lambda *a, **k: None
    sys.modules["prometheus_client"] = prom


def _stub_presidio():
    """Stub presidio so PII tests use regex fallback."""
    for mod in [
        "presidio_analyzer",
        "presidio_analyzer.nlp_engine",
        "presidio_anonymizer",
        "presidio_anonymizer.entities",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)


def _stub_pydantic_settings():
    """Stub pydantic-settings if not installed."""
    try:
        import pydantic_settings  # noqa: F401
    except ImportError:
        ps = types.ModuleType("pydantic_settings")

        class BaseSettings:
            def __init__(self, **k): pass

        ps.BaseSettings = BaseSettings
        sys.modules["pydantic_settings"] = ps


# Apply stubs immediately at import time
_stub_prometheus()
_stub_presidio()
_stub_pydantic_settings()


# ── Re-usable fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf() -> bytes:
    """Minimal valid PDF bytes."""
    return b"%PDF-1.4\n%Fake PDF content for testing\n%%EOF"


@pytest.fixture
def malicious_pdf() -> bytes:
    """PDF with embedded JavaScript action."""
    return b"%PDF-1.4\n/JavaScript alert('xss')\n%%EOF"


@pytest.fixture
def eicar_content() -> bytes:
    """EICAR antivirus test signature."""
    return b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE"


@pytest.fixture
def pe_header() -> bytes:
    """Windows PE (MZ) executable header."""
    return b"\x4d\x5a\x90\x00" * 20


@pytest.fixture
def png_bytes() -> bytes:
    """Minimal PNG magic bytes."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 60


@pytest.fixture
def zip_bytes() -> bytes:
    """Small valid zip file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test.txt", "Hello world")
    return buf.getvalue()


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def uploaded_file_factory():
    """Factory for creating UploadedFile instances."""
    from app.security.file_validator import UploadedFile

    def _make(
        filename: str = "test.pdf",
        content: bytes = b"%PDF-1.4\n%%EOF",
        content_type: str = "application/pdf",
    ) -> UploadedFile:
        return UploadedFile(
            filename=filename,
            content_type=content_type,
            content=content,
        )

    return _make


@pytest.fixture
def pii_detector():
    """PIIDetector using regex fallback (no Presidio required)."""
    from app.security.pii_detector import PIIDetector
    return PIIDetector()


@pytest.fixture
def data_masker():
    """DataMasker with default strategy map."""
    from app.security.data_masker import DataMasker
    return DataMasker()


@pytest.fixture
def injection_detector():
    """PromptInjectionDetector with default 0.7 threshold."""
    from app.security.prompt_injection import PromptInjectionDetector
    return PromptInjectionDetector(risk_threshold=0.7)


@pytest.fixture
def file_validator():
    """FileValidator with default settings."""
    from app.security.file_validator import FileValidator
    return FileValidator()


@pytest.fixture
def security_pipeline():
    """SecurityPipeline in mask mode (not block-on-PII)."""
    from app.security.security_pipeline import SecurityPipeline
    return SecurityPipeline(injection_threshold=0.7, block_on_pii=False)


@pytest.fixture
def security_pipeline_strict():
    """SecurityPipeline in block-on-PII mode."""
    from app.security.security_pipeline import SecurityPipeline
    return SecurityPipeline(injection_threshold=0.7, block_on_pii=True)


@pytest.fixture
def security_request_factory(session_id):
    """Factory for SecurityRequest objects."""
    from app.security.security_pipeline import SecurityRequest

    def _make(query: str = "What is RAG?", files=None) -> SecurityRequest:
        return SecurityRequest(
            query=query,
            uploaded_files=files or [],
            session_id=session_id,
        )

    return _make


@pytest.fixture
def mock_retrieved_chunks():
    """List of realistic RetrievedChunk objects for testing."""
    from app.models.schemas import RetrievedChunk

    return [
        RetrievedChunk(
            chunk_id=f"chunk_{i:04d}",
            document_id="doc_001",
            text=f"This is the content of chunk {i}. It contains relevant information about the topic.",
            score=0.95 - i * 0.05,
            source=f"document_{i % 3 + 1}.pdf",
            page=i + 1,
            metadata={"author": "Test Author", "year": 2024},
            retrieval_method="fused",
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_query_plan():
    """A realistic QueryPlan for SIMPLE query type."""
    from app.models.schemas import QueryPlan, QueryType, RetrievalStrategy

    return QueryPlan(
        query_type=QueryType.SIMPLE,
        retrieval_strategy=RetrievalStrategy.HYBRID,
        use_graph=False,
        require_verification=False,
        verification_level="standard",
        max_hops=1,
        metadata_filters={},
        expanded_queries=["What is RAG?", "retrieval augmented generation overview"],
    )

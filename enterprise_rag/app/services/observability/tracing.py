"""
app/services/observability/tracing.py
LangSmith tracing helpers + async context manager for per-phase spans.
"""
from __future__ import annotations
import os
import time
import functools
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from app.core.config import get_settings
from app.services.observability.logger import get_logger

_settings = get_settings()
_log = get_logger(__name__)


def _configure_langsmith() -> None:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", _settings.langchain_tracing_v2)
    os.environ.setdefault("LANGSMITH_API_KEY",    _settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT",    _settings.langsmith_project)


_configure_langsmith()


@asynccontextmanager
async def trace_span(
    name: str,
    trace_id: str,
    metadata: dict | None = None,
) -> AsyncGenerator[dict, None]:
    """Async context manager that records a named span with timing."""
    start = time.perf_counter()
    span_meta = {"trace_id": trace_id, "span": name, **(metadata or {})}
    _log.info("span_start", **span_meta)
    try:
        yield span_meta
    except Exception as exc:
        _log.error("span_error", error=str(exc), **span_meta)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _log.info("span_end", latency_ms=round(elapsed_ms, 2), **span_meta)


def traced(phase: str) -> Callable:
    """Decorator for async service methods – adds span + latency log."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, trace_id: str = "", **kwargs):
            async with trace_span(f"{phase}.{fn.__name__}", trace_id):
                return await fn(*args, trace_id=trace_id, **kwargs)
        return wrapper
    return decorator

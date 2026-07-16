"""
LangSmith tracing wrapper.
Each pipeline phase wraps its async work in a traced span.
Falls back gracefully when LANGCHAIN_TRACING_V2=false.
"""
from __future__ import annotations

import asyncio
import functools
import os
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Callable, Generator

from app.observability.logger import get_logger, set_context

log = get_logger(__name__)

_ls_enabled = False
_ls_client = None


def init_langsmith(
    api_key: str,
    project: str,
    endpoint: str,
    enabled: bool,
) -> None:
    global _ls_enabled, _ls_client
    if not enabled or not api_key:
        log.info("LangSmith tracing disabled")
        return
    try:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
        from langsmith import Client
        _ls_client = Client()
        _ls_enabled = True
        log.info("LangSmith tracing enabled", extra={"project": project})
    except ImportError:
        log.warning("langsmith package not installed — tracing disabled")


@asynccontextmanager
async def trace_span(
    name: str,
    inputs: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Async context manager that wraps a pipeline phase in a LangSmith run.
    Yields the run_id so callers can attach child spans.
    """
    rid = run_id or str(uuid.uuid4())
    set_context(span=name, trace_id=rid)

    if not _ls_enabled or _ls_client is None:
        yield rid
        return

    try:
        from langsmith import traceable
        # When tracing is live, create a run object
        run = await asyncio.to_thread(
            _ls_client.create_run,
            name=name,
            run_type="chain",
            inputs=inputs or {},
            id=rid,
        )
        try:
            yield rid
            await asyncio.to_thread(
                _ls_client.update_run, rid, end_time=None, outputs={}
            )
        except Exception as exc:
            await asyncio.to_thread(
                _ls_client.update_run, rid, error=str(exc)
            )
            raise
    except Exception:
        # If tracing itself fails, don't break the pipeline
        yield rid


def traced(phase: str) -> Callable:
    """
    Decorator: wraps an async function in a trace span automatically.

    Usage:
        @traced("hybrid_retrieval")
        async def retrieve(...): ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with trace_span(phase):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator

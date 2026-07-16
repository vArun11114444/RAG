"""
Structured JSON logger with request-scoped context via contextvars.
Every log line carries trace_id, phase, and latency automatically.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

_ctx: ContextVar[dict[str, Any]] = ContextVar("log_ctx", default={})


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = _ctx.get({})
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **ctx,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in (
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "module", "msecs", "message", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            ):
                payload[key] = val
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_context(**kwargs: Any) -> None:
    """Attach key-value pairs to the current async context."""
    ctx = dict(_ctx.get({}))
    ctx.update(kwargs)
    _ctx.set(ctx)


def clear_context() -> None:
    _ctx.set({})


class Timer:
    """Context manager that records elapsed ms into a latency dict."""
    def __init__(self, label: str, store: dict[str, float]) -> None:
        self._label = label
        self._store = store

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self._store[self._label] = round(
            (time.perf_counter() - self._start) * 1000, 2
        )

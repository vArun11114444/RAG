"""
app/services/observability/logger.py
Structured JSON logger via structlog.  Attach trace_id to every log line.
"""
import logging
import sys
import structlog
from app.core.config import get_settings

_settings = get_settings()


def configure_logging() -> None:
    log_level = getattr(logging, _settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)

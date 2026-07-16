"""
Security exception hierarchy.

All security violations raise a subclass of SecurityException so the
pipeline can catch a single type and map it to the right HTTP status.
Every exception carries a structured payload for audit logging.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class SecurityEventType(str, Enum):
    PII_DETECTED          = "pii_detected"
    PROMPT_INJECTION      = "prompt_injection"
    FILE_REJECTED         = "file_rejected"
    FILE_TOO_LARGE        = "file_too_large"
    INVALID_MIME          = "invalid_mime"
    MALICIOUS_FILE        = "malicious_file"
    QUERY_BLOCKED         = "query_blocked"
    RATE_LIMIT_EXCEEDED   = "rate_limit_exceeded"


class SecurityException(Exception):
    """
    Base class for all security violations.
    Carries a machine-readable event_type for metrics/alerting,
    a human-readable message, and arbitrary structured context.
    """

    def __init__(
        self,
        message: str,
        event_type: SecurityEventType,
        context: dict[str, Any] | None = None,
        risk_score: float = 1.0,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.event_type = event_type
        self.context = context or {}
        self.risk_score = risk_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.event_type.value,
            "message": self.message,
            "risk_score": self.risk_score,
            "context": self.context,
        }


class PIIDetectedException(SecurityException):
    """Raised when PII is found and masking is disabled or masking fails."""

    def __init__(self, entities: list[dict[str, Any]], message: str = "PII detected in input") -> None:
        super().__init__(
            message=message,
            event_type=SecurityEventType.PII_DETECTED,
            context={"entities": entities},
            risk_score=0.8,
        )
        self.entities = entities


class PromptInjectionException(SecurityException):
    """Raised when prompt injection risk exceeds the configured threshold."""

    def __init__(self, risk_score: float, patterns: list[str]) -> None:
        super().__init__(
            message=f"Prompt injection detected (risk={risk_score:.2f})",
            event_type=SecurityEventType.PROMPT_INJECTION,
            context={"matched_patterns": patterns},
            risk_score=risk_score,
        )
        self.patterns = patterns


class FileValidationException(SecurityException):
    """Raised when an uploaded file fails any validation check."""

    def __init__(
        self,
        filename: str,
        reason: str,
        event_type: SecurityEventType = SecurityEventType.FILE_REJECTED,
    ) -> None:
        super().__init__(
            message=f"File '{filename}' rejected: {reason}",
            event_type=event_type,
            context={"filename": filename, "reason": reason},
            risk_score=0.9,
        )
        self.filename = filename
        self.reason = reason


class MaliciousFileException(FileValidationException):
    """Raised when a file contains known malicious signatures."""

    def __init__(self, filename: str, signature: str) -> None:
        super().__init__(
            filename=filename,
            reason=f"Malicious signature detected: {signature}",
            event_type=SecurityEventType.MALICIOUS_FILE,
        )
        self.risk_score = 1.0
        self.signature = signature

"""
Security Pipeline — app/security/security_pipeline.py

Single entry point for all security checks.
Runs in the following order so cheap checks fail fast:

  1. File validation      (each uploaded file)
  2. Prompt injection     (synchronous, regex-based)
  3. PII detection        (async, Presidio or regex fallback)
  4. PII masking          (if PII found; replaces text before retrieval)

Input:   SecurityRequest  { query, uploaded_files }
Output:  SecurityResult   { sanitized_query, masked_entities, injection_result,
                            file_results, security_score, blocked,
                            block_reason, audit_id }

Blocking policy (configurable via Settings):
  - injection risk ≥ SECURITY_INJECTION_THRESHOLD → block
  - PII found AND SECURITY_BLOCK_ON_PII=True       → block (else mask + continue)
  - any file fails validation                       → block that file

Integrates with existing observability layer:
  - Structured logs via get_logger
  - Prometheus metrics (SECURITY_* counters/histograms)
  - Per-phase latency via Timer
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.observability.logger import Timer, get_logger, set_context
from app.observability.metrics import ERRORS, PHASE_LATENCY
from app.security.data_masker import DataMasker, MaskingResult
from app.security.exceptions import (
    FileValidationException,
    MaliciousFileException,
    PromptInjectionException,
    SecurityException,
    SecurityEventType,
)
from app.security.file_validator import FileValidator, FileValidationResult, UploadedFile
from app.security.metrics import (
    SECURITY_CHECKS_TOTAL,
    SECURITY_BLOCKS_TOTAL,
    SECURITY_INJECTION_RISK,
    SECURITY_PII_ENTITIES,
    SECURITY_FILE_REJECTIONS,
    SECURITY_LATENCY,
)
from app.security.pii_detector import PIIDetectionResult, PIIDetector
from app.security.prompt_injection import InjectionDetectionResult, PromptInjectionDetector

log = get_logger(__name__)


# ── Request / Result schemas ──────────────────────────────────────────────────

@dataclass
class SecurityRequest:
    """Input to the security pipeline."""
    query: str
    uploaded_files: list[UploadedFile] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityResult:
    """Output of the security pipeline. Passed to the RAG planner."""
    # Core outputs
    sanitized_query: str          # query after PII masking (may equal original)
    session_id: str

    # Per-check results
    pii_result: PIIDetectionResult | None          = None
    masking_result: MaskingResult | None           = None
    injection_result: InjectionDetectionResult | None = None
    file_results: list[FileValidationResult]       = field(default_factory=list)

    # Aggregate scores
    security_score: float = 1.0          # 0=dangerous, 1=clean
    injection_risk: float = 0.0
    pii_entity_count: int = 0

    # Decision
    blocked: bool = False
    block_reason: str | None = None
    block_event_type: SecurityEventType | None = None

    # Audit
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_audit_log(self) -> dict[str, Any]:
        """Structured payload for the security audit trail."""
        return {
            "audit_id": self.audit_id,
            "session_id": self.session_id,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "block_event_type": self.block_event_type.value if self.block_event_type else None,
            "security_score": self.security_score,
            "injection_risk": self.injection_risk,
            "pii_entity_count": self.pii_entity_count,
            "pii_types": list({
                e.entity_type
                for e in (self.pii_result.entities if self.pii_result else [])
            }),
            "masked_entities": [
                e.to_dict()
                for e in (self.masking_result.masked_entities if self.masking_result else [])
            ],
            "injection_matches": [
                m.to_dict()
                for m in (self.injection_result.matches if self.injection_result else [])
            ],
            "files_validated": [
                {"filename": r.filename, "size": r.file_size, "mime": r.detected_mime, "warnings": r.warnings}
                for r in self.file_results
            ],
            "latency_ms": self.latency_ms,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SecurityPipeline:
    """
    Orchestrates PII detection, prompt injection detection, file validation,
    and data masking. Thread-safe; one instance per application lifetime.
    """

    def __init__(
        self,
        pii_detector: PIIDetector | None = None,
        data_masker: DataMasker | None = None,
        injection_detector: PromptInjectionDetector | None = None,
        file_validator: FileValidator | None = None,
        injection_threshold: float = 0.7,
        block_on_pii: bool = False,       # False = mask and continue; True = block
        min_pii_score: float = 0.5,       # minimum Presidio confidence to act on
    ) -> None:
        self._pii = pii_detector or PIIDetector(score_threshold=min_pii_score)
        self._masker = data_masker or DataMasker()
        self._injection = injection_detector or PromptInjectionDetector(
            risk_threshold=injection_threshold
        )
        self._file_validator = file_validator or FileValidator()
        self._injection_threshold = injection_threshold
        self._block_on_pii = block_on_pii

    def initialise(self) -> None:
        """Build Presidio engine. Call once at FastAPI startup."""
        self._pii.initialise()
        log.info("security_pipeline_ready")

    async def run(self, request: SecurityRequest) -> SecurityResult:
        """
        Run all security checks in sequence.
        Never raises SecurityException — violations set blocked=True instead.
        Other unexpected exceptions propagate normally.
        """
        set_context(security_session=request.session_id)
        latency: dict[str, float] = {}
        t_total = time.perf_counter()

        SECURITY_CHECKS_TOTAL.inc()

        result = SecurityResult(
            sanitized_query=request.query,
            session_id=request.session_id,
        )

        try:
            # ── Step 1: File validation ────────────────────────────────────
            if request.uploaded_files:
                with Timer("file_validation", latency):
                    result.file_results = await self._validate_files(
                        request.uploaded_files, result
                    )
                if result.blocked:
                    return self._finalise(result, latency, t_total)

            # ── Step 2: Prompt injection detection ─────────────────────────
            with Timer("injection_detection", latency):
                injection_result = await self._detect_injection(request.query, result)
                result.injection_result = injection_result
                result.injection_risk = injection_result.risk_score
                SECURITY_INJECTION_RISK.observe(injection_result.risk_score)

            if result.blocked:
                return self._finalise(result, latency, t_total)

            # ── Step 3: PII detection ──────────────────────────────────────
            with Timer("pii_detection", latency):
                pii_result = await self._pii.detect(request.query)
                result.pii_result = pii_result
                result.pii_entity_count = len(pii_result.entities)
                SECURITY_PII_ENTITIES.observe(len(pii_result.entities))

            # ── Step 4: PII masking or blocking ───────────────────────────
            if pii_result.has_pii:
                with Timer("pii_masking", latency):
                    await self._handle_pii(pii_result, result, request.session_id)

            # ── Compute aggregate security score ───────────────────────────
            result.security_score = self._compute_score(result)

        except SecurityException as exc:
            # Already handled inside sub-methods — shouldn't reach here
            # but acts as final safety net
            self._block(result, exc.message, exc.event_type)
        except Exception:
            ERRORS.labels(phase="security", error_type="unexpected").inc()
            raise

        return self._finalise(result, latency, t_total)

    # ── Step implementations ──────────────────────────────────────────────────

    async def _validate_files(
        self,
        files: list[UploadedFile],
        result: SecurityResult,
    ) -> list[FileValidationResult]:
        """Validate each file. Block on first failure."""
        file_results: list[FileValidationResult] = []
        for uploaded in files:
            try:
                fv = await self._file_validator.validate(uploaded)
                file_results.append(fv)
            except MaliciousFileException as exc:
                SECURITY_FILE_REJECTIONS.labels(reason="malicious").inc()
                log.error(
                    "malicious_file_blocked",
                    extra={
                        "file_name": exc.filename,
                        "signature": exc.signature,
                        "session_id": result.session_id,
                    },
                )
                self._block(result, exc.message, exc.event_type)
                return file_results
            except FileValidationException as exc:
                SECURITY_FILE_REJECTIONS.labels(reason=exc.event_type.value).inc()
                log.warning(
                    "file_rejected",
                    extra={
                        "file_name": exc.filename,
                        "reason": exc.reason,
                        "session_id": result.session_id,
                    },
                )
                self._block(result, exc.message, exc.event_type)
                return file_results
        return file_results

    async def _detect_injection(
        self,
        query: str,
        result: SecurityResult,
    ) -> InjectionDetectionResult:
        """Run injection detection; block if risk exceeds threshold."""
        injection_result = self._injection.detect(query)

        if injection_result.is_injection:
            SECURITY_BLOCKS_TOTAL.labels(reason="injection").inc()
            log.warning(
                "injection_blocked",
                extra={
                    "risk_score": injection_result.risk_score,
                    "categories": injection_result.categories,
                    "session_id": result.session_id,
                },
            )
            self._block(
                result,
                f"Prompt injection detected (risk={injection_result.risk_score:.2f}). "
                f"Categories: {', '.join(injection_result.categories)}",
                SecurityEventType.PROMPT_INJECTION,
            )

        return injection_result

    async def _handle_pii(
        self,
        pii_result: PIIDetectionResult,
        result: SecurityResult,
        session_id: str,
    ) -> None:
        """Either block or mask PII, depending on configuration."""
        if self._block_on_pii:
            SECURITY_BLOCKS_TOTAL.labels(reason="pii").inc()
            entity_types = list({e.entity_type for e in pii_result.entities})
            log.warning(
                "pii_blocked",
                extra={
                    "entity_types": entity_types,
                    "count": len(pii_result.entities),
                    "session_id": session_id,
                },
            )
            self._block(
                result,
                f"Query contains PII: {', '.join(entity_types)}",
                SecurityEventType.PII_DETECTED,
            )
            return

        # Mask and continue
        masking_result = self._masker.mask(pii_result, session_id=session_id)
        result.masking_result = masking_result
        result.sanitized_query = masking_result.masked_text

        log.info(
            "pii_masked_and_continued",
            extra=masking_result.audit_payload(),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _block(
        result: SecurityResult,
        reason: str,
        event_type: SecurityEventType,
    ) -> None:
        result.blocked = True
        result.block_reason = reason
        result.block_event_type = event_type
        result.security_score = 0.0

    def _compute_score(self, result: SecurityResult) -> float:
        """
        Composite security score in [0, 1].
        1.0 = clean; 0.0 = high risk / blocked.
        """
        if result.blocked:
            return 0.0

        injection_penalty = result.injection_risk * 0.5
        pii_penalty = min(0.3, result.pii_entity_count * 0.05)
        score = round(max(0.0, 1.0 - injection_penalty - pii_penalty), 3)
        return score

    def _finalise(
        self,
        result: SecurityResult,
        latency: dict[str, float],
        t_total: float,
    ) -> SecurityResult:
        latency["security_total"] = round((time.perf_counter() - t_total) * 1000, 2)
        result.latency_ms = latency

        SECURITY_LATENCY.observe(latency["security_total"] / 1000)
        PHASE_LATENCY.labels(phase="security").observe(latency["security_total"] / 1000)

        # Emit full audit log
        log.info("security_audit", extra=result.to_audit_log())

        if result.blocked:
            SECURITY_BLOCKS_TOTAL.labels(
                reason=result.block_event_type.value if result.block_event_type else "unknown"
            ).inc()

        return result

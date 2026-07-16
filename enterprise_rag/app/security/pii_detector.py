"""
PII Detector — app/security/pii_detector.py

Uses Microsoft Presidio for entity recognition.
Adds custom recognizers for Indian PII:
  - Aadhaar numbers  (12-digit, space/dash separated)
  - PAN numbers      (AAAAA9999A format)

Supported built-in entities:
  EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, IP_ADDRESS,
  PERSON, LOCATION, IBAN_CODE, US_SSN, URL

Falls back gracefully when Presidio is not installed — logs a warning
and returns an empty entity list so the pipeline continues unblocked.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from app.observability.logger import get_logger

log = get_logger(__name__)

# ── Target entity types ───────────────────────────────────────────────────────
DEFAULT_ENTITIES: list[str] = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "PERSON",
    "IBAN_CODE",
    "URL",
    "IN_AADHAAR",
    "IN_PAN",
]


@dataclass
class PIIEntity:
    """Single detected PII span."""
    entity_type: str
    text: str                    # original text (may be empty if anonymised)
    start: int
    end: int
    score: float                 # Presidio confidence 0–1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class PIIDetectionResult:
    """Result of a single PII scan."""
    original_text: str
    entities: list[PIIEntity] = field(default_factory=list)
    has_pii: bool = False
    presidio_available: bool = True

    @property
    def entity_types(self) -> list[str]:
        return list({e.entity_type for e in self.entities})

    @property
    def max_score(self) -> float:
        return max((e.score for e in self.entities), default=0.0)


# ── Custom Presidio recognizers ───────────────────────────────────────────────

def _build_aadhaar_recognizer() -> Any:
    """
    Aadhaar: 12 digits, optionally separated by spaces or dashes.
    Examples: 1234 5678 9012 | 1234-5678-9012 | 123456789012
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    patterns = [
        Pattern(
            name="aadhaar_spaced",
            regex=r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b",
            score=0.85,
        ),
        Pattern(
            name="aadhaar_plain",
            regex=r"\b[2-9]\d{11}\b",   # first digit 2-9 per UIDAI spec
            score=0.6,
        ),
    ]
    return PatternRecognizer(
        supported_entity="IN_AADHAAR",
        patterns=patterns,
        name="AadhaarRecognizer",
    )


def _build_pan_recognizer() -> Any:
    """
    PAN: 5 alpha + 4 digits + 1 alpha (e.g. ABCDE1234F)
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    return PatternRecognizer(
        supported_entity="IN_PAN",
        patterns=[
            Pattern(
                name="pan_standard",
                regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                score=0.9,
            )
        ],
        name="PANRecognizer",
    )


# ── Detector ──────────────────────────────────────────────────────────────────

class PIIDetector:
    """
    Async-safe wrapper around Presidio AnalyzerEngine.
    Engine is built once and reused; Presidio is CPU-bound so
    analysis is offloaded to a thread pool via asyncio.to_thread.
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        score_threshold: float = 0.4,
        language: str = "en",
    ) -> None:
        self._entities = entities or DEFAULT_ENTITIES
        self._score_threshold = score_threshold
        self._language = language
        self._engine: Any = None
        self._available = False

    def initialise(self) -> None:
        """Build the Presidio engine. Call once at application startup."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Use spaCy small model — already in requirements for the KG phase
            provider = NlpEngineProvider(nlp_engine_name="spacy")
            nlp_engine = provider.create_engine()

            engine = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
            engine.registry.add_recognizer(_build_aadhaar_recognizer())
            engine.registry.add_recognizer(_build_pan_recognizer())

            self._engine = engine
            self._available = True
            log.info("presidio_initialised", extra={"entities": self._entities})

        except ImportError:
            log.warning(
                "presidio_not_installed",
                extra={"msg": "Install presidio-analyzer for PII detection. Falling back to regex-only."},
            )
            self._available = False
        except Exception as exc:
            log.warning("presidio_init_failed", extra={"error": str(exc)})
            self._available = False

    async def detect(self, text: str) -> PIIDetectionResult:
        """
        Detect PII in *text*.
        Returns PIIDetectionResult — never raises.
        """
        if not text or not text.strip():
            return PIIDetectionResult(original_text=text)

        if self._available and self._engine is not None:
            try:
                entities = await asyncio.to_thread(self._presidio_detect, text)
                return PIIDetectionResult(
                    original_text=text,
                    entities=entities,
                    has_pii=bool(entities),
                    presidio_available=True,
                )
            except Exception as exc:
                log.warning("presidio_detect_failed", extra={"error": str(exc)})

        # ── Regex fallback ─────────────────────────────────────────────────
        entities = await asyncio.to_thread(self._regex_detect, text)
        return PIIDetectionResult(
            original_text=text,
            entities=entities,
            has_pii=bool(entities),
            presidio_available=False,
        )

    def _presidio_detect(self, text: str) -> list[PIIEntity]:
        results = self._engine.analyze(
            text=text,
            entities=self._entities,
            language=self._language,
            score_threshold=self._score_threshold,
        )
        entities: list[PIIEntity] = []
        for r in results:
            span_text = text[r.start:r.end]
            entities.append(PIIEntity(
                entity_type=r.entity_type,
                text=span_text,
                start=r.start,
                end=r.end,
                score=r.score,
                metadata={"recognizer": r.recognition_metadata.get("recognizer_name", "") if r.recognition_metadata else ""},
            ))
        # Sort by position
        entities.sort(key=lambda e: e.start)
        return entities

    # ── Regex fallback patterns ───────────────────────────────────────────────
    _REGEX_PATTERNS: list[tuple[str, str, float]] = [
        ("EMAIL_ADDRESS",  r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",      0.95),
        ("PHONE_NUMBER",   r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b|\b\+?1?\s?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b", 0.8),
        ("CREDIT_CARD",    r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))\s?\d{4}\s?\d{4}\s?\d{4}\b", 0.9),
        ("IP_ADDRESS",     r"\b(?:\d{1,3}\.){3}\d{1,3}\b",                                0.85),
        ("IN_AADHAAR",     r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b",                            0.85),
        ("IN_PAN",         r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",                                  0.9),
    ]

    def _regex_detect(self, text: str) -> list[PIIEntity]:
        entities: list[PIIEntity] = []
        for entity_type, pattern, score in self._REGEX_PATTERNS:
            for m in re.finditer(pattern, text):
                entities.append(PIIEntity(
                    entity_type=entity_type,
                    text=m.group(),
                    start=m.start(),
                    end=m.end(),
                    score=score,
                    metadata={"recognizer": "regex_fallback"},
                ))
        entities.sort(key=lambda e: e.start)
        return entities

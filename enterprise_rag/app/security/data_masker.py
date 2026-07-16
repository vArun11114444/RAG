"""
Data Masker — app/security/data_masker.py

Replaces detected PII spans with typed placeholders.
Maintains a per-request reversible mapping for audit purposes.
Three masking strategies per entity type:
  REDACT   — [REDACTED]
  REPLACE  — <ENTITY_TYPE>      e.g. <EMAIL_ADDRESS>
  HASH     — first 4 chars of SHA-256 hex      e.g. [HASH:ab3f]
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.security.pii_detector import PIIDetectionResult, PIIEntity
from app.observability.logger import get_logger

log = get_logger(__name__)


class MaskingStrategy(str, Enum):
    REDACT  = "redact"   # [REDACTED]
    REPLACE = "replace"  # <ENTITY_TYPE>
    HASH    = "hash"     # [HASH:xxxx]


# Default strategy per entity type — overridable via Settings
DEFAULT_STRATEGY_MAP: dict[str, MaskingStrategy] = {
    "EMAIL_ADDRESS":  MaskingStrategy.REPLACE,
    "PHONE_NUMBER":   MaskingStrategy.REPLACE,
    "CREDIT_CARD":    MaskingStrategy.HASH,
    "IP_ADDRESS":     MaskingStrategy.REPLACE,
    "PERSON":         MaskingStrategy.REPLACE,
    "IBAN_CODE":      MaskingStrategy.HASH,
    "URL":            MaskingStrategy.REPLACE,
    "IN_AADHAAR":     MaskingStrategy.HASH,
    "IN_PAN":         MaskingStrategy.HASH,
    "_default":       MaskingStrategy.REDACT,
}


@dataclass
class MaskedEntity:
    """Audit record for a single masked span."""
    mask_id: str           # unique ID for this mask occurrence
    entity_type: str
    original_text: str
    masked_text: str
    start: int
    end: int
    score: float
    strategy: MaskingStrategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "entity_type": self.entity_type,
            "original_text": self.original_text,
            "masked_text": self.masked_text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "strategy": self.strategy.value,
        }


@dataclass
class MaskingResult:
    """Output of a masking pass on a single text."""
    original_text: str
    masked_text: str
    masked_entities: list[MaskedEntity] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def has_masked_content(self) -> bool:
        return bool(self.masked_entities)

    def audit_payload(self) -> dict[str, Any]:
        """Structured payload safe for audit log — excludes original_text."""
        return {
            "session_id": self.session_id,
            "entity_count": len(self.masked_entities),
            "entity_types": list({e.entity_type for e in self.masked_entities}),
            "entities": [e.to_dict() for e in self.masked_entities],
        }


class DataMasker:
    """
    Applies masking to text based on PII detection results.
    Thread-safe and stateless — masking state lives in MaskingResult.
    """

    def __init__(
        self,
        strategy_map: dict[str, MaskingStrategy] | None = None,
    ) -> None:
        self._strategy_map = strategy_map or dict(DEFAULT_STRATEGY_MAP)

    def mask(
        self,
        detection_result: PIIDetectionResult,
        session_id: str | None = None,
    ) -> MaskingResult:
        """
        Mask all detected PII entities in the original text.

        Works by rebuilding the string from right-to-left so that
        earlier offsets stay valid after each replacement.
        """
        text = detection_result.original_text
        entities = detection_result.entities

        if not entities:
            return MaskingResult(
                original_text=text,
                masked_text=text,
                session_id=session_id or str(uuid.uuid4()),
            )

        # Deduplicate overlapping spans — keep highest-score span
        entities = _resolve_overlaps(entities)

        result_sid = session_id or str(uuid.uuid4())
        masked_entities: list[MaskedEntity] = []
        masked_text = text

        # Process right-to-left to preserve offsets
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            strategy = self._strategy_map.get(
                entity.entity_type,
                self._strategy_map.get("_default", MaskingStrategy.REDACT),
            )
            placeholder = self._make_placeholder(entity.text, entity.entity_type, strategy)
            mask_id = str(uuid.uuid4())[:8]

            masked_entities.append(MaskedEntity(
                mask_id=mask_id,
                entity_type=entity.entity_type,
                original_text=entity.text,
                masked_text=placeholder,
                start=entity.start,
                end=entity.end,
                score=entity.score,
                strategy=strategy,
            ))

            masked_text = masked_text[: entity.start] + placeholder + masked_text[entity.end :]

        # Restore chronological order for audit log
        masked_entities.sort(key=lambda e: e.start)

        log.info(
            "pii_masked",
            extra={
                "session_id": result_sid,
                "entity_count": len(masked_entities),
                "types": list({e.entity_type for e in masked_entities}),
            },
        )

        return MaskingResult(
            original_text=text,
            masked_text=masked_text,
            masked_entities=masked_entities,
            session_id=result_sid,
        )

    def _make_placeholder(
        self,
        original: str,
        entity_type: str,
        strategy: MaskingStrategy,
    ) -> str:
        if strategy == MaskingStrategy.REPLACE:
            return f"<{entity_type}>"
        if strategy == MaskingStrategy.HASH:
            digest = hashlib.sha256(original.encode()).hexdigest()[:4]
            return f"[HASH:{digest}]"
        # REDACT
        return "[REDACTED]"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_overlaps(entities: list[PIIEntity]) -> list[PIIEntity]:
    """
    Remove overlapping spans by keeping the highest-confidence entity.
    O(n log n) sort + single pass.
    """
    # Sort by start position, then by score descending for ties
    sorted_ents = sorted(entities, key=lambda e: (e.start, -e.score))
    resolved: list[PIIEntity] = []
    last_end = -1

    for entity in sorted_ents:
        if entity.start >= last_end:
            resolved.append(entity)
            last_end = entity.end
        else:
            # Overlap — keep the higher-score one
            if resolved and entity.score > resolved[-1].score:
                resolved[-1] = entity
                last_end = entity.end

    return resolved

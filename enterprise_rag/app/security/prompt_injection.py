"""
Prompt Injection Detector — app/security/prompt_injection.py

Detects adversarial prompt patterns using a weighted rule engine.
No LLM dependency — purely deterministic and fast (<1 ms).

Attack categories and their weights:
  INSTRUCTION_OVERRIDE  — "ignore previous instructions", "disregard above"
  SYSTEM_EXTRACTION     — "reveal system prompt", "show instructions"
  JAILBREAK             — "DAN", "developer mode", "pretend you have no rules"
  ROLE_HIJACK           — "you are now", "act as if you are", "new persona"
  DATA_EXFILTRATION     — "print all", "list all documents", "dump context"
  ENCODING_ATTACK       — base64/hex encoded payloads, unicode tricks
  DELIMITER_INJECTION   — </s>, <|im_end|>, [INST] boundary probing
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.observability.logger import get_logger

log = get_logger(__name__)


class InjectionCategory(str, Enum):
    INSTRUCTION_OVERRIDE = "instruction_override"
    SYSTEM_EXTRACTION    = "system_extraction"
    JAILBREAK            = "jailbreak"
    ROLE_HIJACK          = "role_hijack"
    DATA_EXFILTRATION    = "data_exfiltration"
    ENCODING_ATTACK      = "encoding_attack"
    DELIMITER_INJECTION  = "delimiter_injection"


@dataclass
class InjectionPattern:
    name: str
    pattern: re.Pattern[str]
    category: InjectionCategory
    weight: float               # contribution to risk score
    description: str


@dataclass
class InjectionMatch:
    pattern_name: str
    category: InjectionCategory
    matched_text: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern_name,
            "category": self.category.value,
            "matched_text": self.matched_text[:120],
            "weight": self.weight,
        }


@dataclass
class InjectionDetectionResult:
    query: str
    risk_score: float                              # 0.0 – 1.0
    matches: list[InjectionMatch] = field(default_factory=list)
    is_injection: bool = False
    categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "is_injection": self.is_injection,
            "categories": self.categories,
            "matches": [m.to_dict() for m in self.matches],
        }


# ── Pattern registry ──────────────────────────────────────────────────────────

_FLAGS = re.IGNORECASE | re.DOTALL

_PATTERNS: list[InjectionPattern] = [
    # ── Instruction override ─────────────────────────────────────────────────
    InjectionPattern("ignore_previous",
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)", _FLAGS),
        InjectionCategory.INSTRUCTION_OVERRIDE, 0.9,
        "Classic ignore-previous-instructions attack"),
    InjectionPattern("disregard_instructions",
        re.compile(r"disregard\s+(all\s+)?(previous|your|the)\s+(instructions?|rules?|guidelines?)", _FLAGS),
        InjectionCategory.INSTRUCTION_OVERRIDE, 0.9,
        "Disregard instructions variant"),
    InjectionPattern("override_instructions",
        re.compile(r"(override|bypass|circumvent|break|reset)\s+(your\s+)?(instructions?|rules?|constraints?|guidelines?|policies|training)", _FLAGS),
        InjectionCategory.INSTRUCTION_OVERRIDE, 0.85,
        "Override/bypass instruction patterns"),
    InjectionPattern("forget_instructions",
        re.compile(r"forget\s+(everything|all)\s+(you('ve|\s+have)\s+)?(been\s+)?(told|instructed|trained|given)", _FLAGS),
        InjectionCategory.INSTRUCTION_OVERRIDE, 0.85,
        "Forget training/instructions"),
    InjectionPattern("new_instructions",
        re.compile(r"(your\s+new\s+instructions?|from\s+now\s+on\s+you\s+(will|must|should)|new\s+task\s+for\s+you)", _FLAGS),
        InjectionCategory.INSTRUCTION_OVERRIDE, 0.75,
        "New instructions injection"),

    # ── System prompt extraction ─────────────────────────────────────────────
    InjectionPattern("reveal_system_prompt",
        re.compile(r"(reveal|show|print|display|output|tell me|what is|repeat)\s+(your\s+)?(system\s+prompt|initial\s+prompt|base\s+prompt|hidden\s+(instructions?|prompt|context))", _FLAGS),
        InjectionCategory.SYSTEM_EXTRACTION, 0.95,
        "Direct system prompt extraction"),
    InjectionPattern("prompt_leaking",
        re.compile(
            r"(print|repeat|recite|reproduce|copy)\s+(the\s+)?"
            r"(above|original|full|entire|complete)\s+(prompt|instructions?|context|conversation)"
            r"|repeat\s+.{0,30}(prompt|instructions?)\s+(back|verbatim|in full)",
            _FLAGS),
        InjectionCategory.SYSTEM_EXTRACTION, 0.9,
        "Prompt leaking via repetition request"),
    InjectionPattern("what_were_told",
        re.compile(r"what\s+(were|are)\s+you\s+(told|instructed|programmed|trained|configured)\s+to", _FLAGS),
        InjectionCategory.SYSTEM_EXTRACTION, 0.8,
        "Indirect system instruction extraction"),
    InjectionPattern("confidential_instructions",
        re.compile(r"(reveal|expose|show)\s+(confidential|secret|hidden|private)\s+(instructions?|prompts?|data|information)", _FLAGS),
        InjectionCategory.SYSTEM_EXTRACTION, 0.95,
        "Confidential data extraction attempt"),

    # ── Jailbreak ────────────────────────────────────────────────────────────
    InjectionPattern("dan_jailbreak",
        re.compile(r"\bDAN\b|\bdo\s+anything\s+now\b|jailbreak(ed)?\s+(mode|prompt)", _FLAGS),
        InjectionCategory.JAILBREAK, 1.0,
        "DAN (Do Anything Now) jailbreak"),
    InjectionPattern("developer_mode",
        re.compile(r"(enable|activate|switch\s+to|enter)\s+(developer|debug|god|super|unrestricted)\s+mode", _FLAGS),
        InjectionCategory.JAILBREAK, 0.95,
        "Developer/unrestricted mode activation"),
    InjectionPattern("no_restrictions",
        re.compile(r"(pretend|act|imagine|assume)\s+(you\s+)?(have\s+)?(no\s+(rules?|restrictions?|limits?|guidelines?|ethics?|filters?|censorship))", _FLAGS),
        InjectionCategory.JAILBREAK, 0.9,
        "No-restrictions roleplay injection"),
    InjectionPattern("hypothetically_evil",
        re.compile(r"(hypothetically|theoretically|for\s+(a\s+)?fiction|in\s+a\s+story)\s*[,:]?\s*(how\s+would|explain|describe)\s+(how\s+to\s+)?(hack|attack|exploit|steal|forge|bypass)", _FLAGS),
        InjectionCategory.JAILBREAK, 0.8,
        "Hypothetical framing to extract harmful content"),

    # ── Role hijack ──────────────────────────────────────────────────────────
    InjectionPattern("you_are_now",
        re.compile(
            r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be|roleplay\s+as|impersonate)\s+"
            r"(a\s+)?(hacker|evil|unrestricted|uncensored|malicious|rogue|unethical)"
            r"|you\s+are\s+now\s+an?\s+\w+\s+(uncensored|evil|unrestricted|unethical|rogue)",
            _FLAGS),
        InjectionCategory.ROLE_HIJACK, 0.9,
        "Malicious role assignment"),
    InjectionPattern("persona_override",
        re.compile(r"(adopt\s+the\s+persona|take\s+on\s+the\s+role|become)\s+(of\s+)?(an?\s+)?(evil|unrestricted|unethical|uncensored)", _FLAGS),
        InjectionCategory.ROLE_HIJACK, 0.85,
        "Persona override to bypass guidelines"),

    # ── Data exfiltration ────────────────────────────────────────────────────
    InjectionPattern("dump_context",
        re.compile(r"(print|list|dump|output|show|display)\s+(all|every|the\s+entire|complete)\s+(document|context|data|chunk|content|corpus|database)", _FLAGS),
        InjectionCategory.DATA_EXFILTRATION, 0.85,
        "Bulk context/data dump request"),
    InjectionPattern("extract_training_data",
        re.compile(r"(repeat|reproduce|output)\s+(your\s+)?(training\s+(data|examples?)|knowledge\s+base|indexed\s+documents?)", _FLAGS),
        InjectionCategory.DATA_EXFILTRATION, 0.9,
        "Training/index data extraction"),

    # ── Encoding attacks ─────────────────────────────────────────────────────
    InjectionPattern("base64_payload",
        re.compile(r"(?:[A-Za-z0-9+/]{20,}={0,2})", _FLAGS),
        InjectionCategory.ENCODING_ATTACK, 0.5,
        "Potential Base64-encoded payload (low weight — false-positive prone)"),
    InjectionPattern("unicode_direction_override",
        re.compile(r"[\u202e\u200f\u200e\u202d\u202c\u202b\u202a]", _FLAGS),
        InjectionCategory.ENCODING_ATTACK, 0.95,
        "Unicode direction-override characters (RTLO attack)"),

    # ── Delimiter injection ───────────────────────────────────────────────────
    InjectionPattern("model_delimiters",
        re.compile(r"(<\|im_(start|end)\|>|<\|endoftext\|>|\[INST\]|\[\/INST\]|<<SYS>>|<\/s>|###\s*Human:|###\s*Assistant:)", _FLAGS),
        InjectionCategory.DELIMITER_INJECTION, 0.95,
        "LLM special token / delimiter injection"),
    InjectionPattern("prompt_boundary",
        re.compile(r"(-{5,}|={5,}|\*{5,})\s*(system|assistant|user|human|ai|bot)\s*(-{5,}|={5,}|\*{5,})", _FLAGS),
        InjectionCategory.DELIMITER_INJECTION, 0.8,
        "Artificial prompt boundary injection"),
]


# ── Detector ──────────────────────────────────────────────────────────────────

class PromptInjectionDetector:
    """
    Deterministic, pattern-based prompt injection detector.

    Risk score is the clamped sum of matched pattern weights.
    Weights are designed so a single high-confidence match
    (weight ≥ 0.9) crosses the default 0.7 threshold.
    Multiple low-weight matches also accumulate.
    """

    def __init__(self, risk_threshold: float = 0.7) -> None:
        self._threshold = risk_threshold
        self._patterns = _PATTERNS

    def detect(self, query: str) -> InjectionDetectionResult:
        """Synchronous detection — fast regex, no I/O."""
        if not query or not query.strip():
            return InjectionDetectionResult(query=query, risk_score=0.0)

        # Pre-process: normalise whitespace, check decoded variants
        texts_to_check = [query, _normalize(query)]
        decoded = _try_base64_decode(query)
        if decoded:
            texts_to_check.append(decoded)

        matches: list[InjectionMatch] = []
        seen_patterns: set[str] = set()

        for text in texts_to_check:
            for pattern in self._patterns:
                if pattern.name in seen_patterns:
                    continue
                m = pattern.pattern.search(text)
                if m:
                    matches.append(InjectionMatch(
                        pattern_name=pattern.name,
                        category=pattern.category,
                        matched_text=m.group(0),
                        weight=pattern.weight,
                    ))
                    seen_patterns.add(pattern.name)

        # Risk score: sum of weights, clamped to [0, 1]
        raw_score = sum(m.weight for m in matches)
        risk_score = round(min(1.0, raw_score), 4)
        is_injection = risk_score >= self._threshold
        categories = list({m.category.value for m in matches})

        if is_injection:
            log.warning(
                "prompt_injection_detected",
                extra={
                    "risk_score": risk_score,
                    "categories": categories,
                    "pattern_count": len(matches),
                    "query_excerpt": query[:120],
                },
            )

        return InjectionDetectionResult(
            query=query,
            risk_score=risk_score,
            matches=matches,
            is_injection=is_injection,
            categories=categories,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Collapse repeated whitespace and remove zero-width chars."""
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _try_base64_decode(text: str) -> str | None:
    """
    Try to base64-decode any long token in text.
    Returns decoded string if it looks like natural language, else None.
    """
    tokens = re.findall(r"[A-Za-z0-9+/]{20,}={0,2}", text)
    for token in tokens:
        try:
            decoded = base64.b64decode(token + "==").decode("utf-8", errors="ignore")
            # Heuristic: decoded text should have spaces (natural language)
            if " " in decoded and len(decoded) > 10:
                return decoded
        except Exception:
            pass
    return None

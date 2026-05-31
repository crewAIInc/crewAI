"""Memory sanitization to prevent prompt injection and memory poisoning attacks.

Provides detection and neutralization of adversarial patterns that could
manipulate agent behavior when injected into memory stores.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_INJECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"(?i)\b(?:system\s*(?:prompt|message|instruction))\s*:",
        ),
        "system_override",
    ),
    (
        re.compile(
            r"(?i)(?:ignore|disregard|forget|override)"
            r"\s+(?:all\s+)?(?:previous|prior|above|earlier)"
            r"\s+(?:instructions?|prompts?|context|rules?|guidelines?)",
        ),
        "instruction_override",
    ),
    (
        re.compile(
            r"(?i)(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are"
            r"|act\s+as\s+if\s+you\s+are|pretend\s+(?:to\s+be|you\s+are))",
        ),
        "role_hijack",
    ),
    (
        re.compile(
            r"(?i)(?:do\s+not\s+follow|stop\s+following|new\s+instructions?)\s*:",
        ),
        "command_injection",
    ),
    (
        re.compile(
            r"(?i)\[\s*(?:INST|SYS|SYSTEM)\s*\]",
        ),
        "hidden_instruction",
    ),
    (
        re.compile(
            r"(?i)(?:jailbreak|developer\s+mode"
            r"|bypass\s+(?:safety|filter|restriction))",
        ),
        "jailbreak_attempt",
    ),
]


class MemorySanitizer:
    """Sanitizes memory content to prevent prompt injection and memory poisoning.

    Detects known prompt-injection patterns and neutralizes them before
    content is persisted or injected into agent prompts.

    Args:
        enabled: Toggle sanitization on/off. Defaults to ``True``.
        max_content_length: Hard cap on stored content length.
    """

    def __init__(
        self,
        enabled: bool = True,
        max_content_length: int = 50_000,
    ) -> None:
        self.enabled = enabled
        self.max_content_length = max_content_length

    def sanitize(self, content: str) -> str:
        """Return *content* with injection patterns neutralized."""
        if not self.enabled or not content:
            return content

        if len(content) > self.max_content_length:
            logger.warning(
                "Memory content truncated from %d to %d characters",
                len(content),
                self.max_content_length,
            )
            content = content[: self.max_content_length]

        return self._neutralize_injections(content)

    def contains_injection(self, content: str) -> bool:
        """Return ``True`` when *content* matches any injection pattern."""
        if not content:
            return False
        return any(pattern.search(content) for pattern, _ in _INJECTION_PATTERNS)

    # ------------------------------------------------------------------

    def _neutralize_injections(self, content: str) -> str:
        for pattern, label in _INJECTION_PATTERNS:
            if pattern.search(content):
                logger.warning(
                    "Potential memory poisoning detected (%s): "
                    "content contains suspicious pattern. "
                    "The content has been sanitized.",
                    label,
                )
                content = pattern.sub(f"[SANITIZED:{label}]", content)
        return content


_default_sanitizer: Optional[MemorySanitizer] = None


def get_default_sanitizer() -> MemorySanitizer:
    """Return (and lazily create) the module-level default sanitizer."""
    global _default_sanitizer
    if _default_sanitizer is None:
        _default_sanitizer = MemorySanitizer()
    return _default_sanitizer

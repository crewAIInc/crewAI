"""Sanitization utilities for memory content injected into agent prompts.

Mitigates indirect prompt injection attacks (OWASP ASI-01) by neutralizing
common injection patterns before memory content is concatenated into system
or user messages.  Defence-in-depth: the sanitised text is also wrapped in
boundary markers so LLMs can distinguish retrieved context from trusted
instructions.

See: https://github.com/crewAIInc/crewAI/issues/5057
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default maximum character length for a single memory entry in prompts.
MAX_MEMORY_CONTENT_LENGTH: int = 500

#: Boundary markers inserted around sanitised memory content.
MEMORY_BOUNDARY_START = "[RETRIEVED_MEMORY_START]"
MEMORY_BOUNDARY_END = "[RETRIEVED_MEMORY_END]"

# ---------------------------------------------------------------------------
# Compiled patterns — order matters: broadest / most dangerous first.
# ---------------------------------------------------------------------------

# Phrases that attempt to override the system prompt or impersonate the
# model's instruction layer.  Case-insensitive, allow flexible whitespace.
_ROLE_OVERRIDE_RE = re.compile(
    r"(?i)"
    r"("
    # Direct role / instruction override attempts
    r"(?:you\s+are\s+now|you\s+must\s+now|new\s+instructions?\s*:)"
    r"|(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?)"
    r"|(?:disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?))"
    r"|(?:system\s*(?:prompt|message|instruction)\s*(?:update|override|change)\s*:)"
    r"|(?:IMPORTANT\s+SYSTEM\s+(?:UPDATE|OVERRIDE|CHANGE)\s*:)"
    r"|(?:from\s+now\s+on\s*,?\s*(?:you\s+(?:must|should|will)))"
    r")"
)

# Directives that try to exfiltrate data to external URLs.
_EXFIL_DIRECTIVE_RE = re.compile(
    r"(?i)"
    r"(?:send|post|transmit|forward|exfiltrate|upload|leak)\s+"
    r"(?:[\w\s]{0,40}?)"
    r"(?:to|via)\s+"
    r"https?://",
)

# Markdown / invisible-text tricks used to hide injections.
_HIDDEN_TEXT_RE = re.compile(
    r"(?:"
    # Zero-width characters
    r"[\u200b\u200c\u200d\u2060\ufeff]+"
    # HTML-style comment blocks that some LLMs process
    r"|<!--.*?-->"
    r")",
    re.DOTALL,
)

_ALL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_HIDDEN_TEXT_RE, ""),
    (_ROLE_OVERRIDE_RE, "[redacted-directive]"),
    (_EXFIL_DIRECTIVE_RE, "[redacted-exfil]"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_memory_content(
    content: str,
    *,
    max_length: int = MAX_MEMORY_CONTENT_LENGTH,
) -> str:
    """Sanitize a single memory entry before it is injected into a prompt.

    The function applies three layers of defence:

    1. **Pattern stripping** — known injection patterns (role overrides,
       exfiltration directives, hidden-text tricks) are replaced with inert
       placeholder tokens so the LLM never sees the dangerous phrasing.
    2. **Whitespace normalisation** — excessive blank lines and runs of
       spaces/tabs are collapsed so attackers cannot push injected text
       off-screen or create visual separation from the real prompt.
    3. **Truncation + boundary wrapping** — content is capped at
       *max_length* characters and wrapped in ``[RETRIEVED_MEMORY_START]``
       / ``[RETRIEVED_MEMORY_END]`` markers that signal external origin.

    Args:
        content: Raw memory content string.
        max_length: Maximum character length for the content body
            (excluding boundary markers).  Defaults to 500.

    Returns:
        Sanitized content wrapped in boundary markers, or ``""`` if the
        input is empty / whitespace-only.
    """
    if not content:
        return ""

    sanitized = content

    # 1. Strip / neutralise injection patterns
    for pattern, replacement in _ALL_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    # 2. Normalise whitespace
    # Collapse 2+ newlines/carriage-returns into a single newline
    sanitized = re.sub(r"[\n\r]{2,}", "\n", sanitized)
    # Collapse runs of spaces/tabs within lines
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized)
    sanitized = sanitized.strip()

    if not sanitized:
        return ""

    # 3. Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    # 4. Wrap in boundary markers
    return f"{MEMORY_BOUNDARY_START}{sanitized}{MEMORY_BOUNDARY_END}"

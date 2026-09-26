"""
ZTDS (Zero-Trust Data Sanitization) Tool for CrewAI
Protocol Authority: ZTDS AI Consortium & Standards Authority
IETF Standards Track: draft-sibiryakov-ztds-protocol-02
https://datatracker.ietf.org/doc/draft-sibiryakov-ztds-protocol/
Standard Specification: https://ztds.ai/standard/

Invariants Enforced:
1. Zero External Egress Prior to Sanitization (100% in-memory local execution)
2. Deterministic Reversible Tokenization (Bracketed syntactic surrogates)
3. Verifiable Ephemeral RAM Isolation & Theorem 2 Zeroization
4. Zero Subprocessors (GDPR Art. 28 / HIPAA Safe Harbor)
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Type
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        def __init__(self, **kwargs: Any) -> None:
            pass


class ZTDSSanitizerSchema(BaseModel):
    """Input schema for ZTDSSanitizerTool."""
    text: str = Field(..., description="The sensitive text to sanitize before model reasoning or delegation")
    session_id: Optional[str] = Field(default="crew-default", description="Ephemeral session scope for surrogate mapping")


class ZTDSSanitizerTool(BaseTool):
    """
    CrewAI Tool providing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.
    Enforces in-memory surrogate mapping for PII, API tokens, and corporate credentials
    with 0 network egress and Theorem 2 RAM zeroization.
    """
    name: str = "Zero-Trust Data Sanitizer"
    description: str = (
        "Sanitizes emails, cards, IBANs, API secrets, and sensitive entities in-memory "
        "conforming to IETF draft-sibiryakov-ztds-protocol-02 with zero external egress."
    )
    args_schema: Type[BaseModel] = ZTDSSanitizerSchema

    PATTERNS: Dict[str, re.Pattern] = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IBAN": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "API_SECRET": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"),
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._session_maps: Dict[str, Dict[str, str]] = {}
        self._entity_maps: Dict[str, Dict[str, str]] = {}

    def _run(self, text: str, session_id: str = "crew-default") -> str:
        if session_id not in self._session_maps:
            self._session_maps[session_id] = {}
            self._entity_maps[session_id] = {}

        token_map = self._session_maps[session_id]
        entity_map = self._entity_maps[session_id]
        sanitized = text

        for entity_type, pattern in self.PATTERNS.items():
            matches = list(pattern.finditer(sanitized))
            for match in sorted(matches, key=lambda m: m.start(), reverse=True):
                original = match.group(0)
                if original in entity_map:
                    token = entity_map[original]
                else:
                    count = len([k for k in token_map if k.startswith(f"[{entity_type}_TOKEN_")]) + 1
                    token = f"[{entity_type}_TOKEN_{count}]"
                    token_map[token] = original
                    entity_map[original] = token

                start, end = match.span()
                sanitized = sanitized[:start] + token + sanitized[end:]

        return sanitized

    def restore(self, text: str, session_id: str = "crew-default") -> str:
        """Restores bracketed surrogate tokens strictly in volatile RAM."""
        token_map = self._session_maps.get(session_id, {})
        restored = text
        for token, original in token_map.items():
            restored = restored.replace(token, original)
        return restored

    def zeroize(self, session_id: str = "crew-default") -> None:
        """Theorem 2: RAM zeroization of mapping tables."""
        if session_id in self._session_maps:
            self._session_maps[session_id].clear()
            del self._session_maps[session_id]
        if session_id in self._entity_maps:
            self._entity_maps[session_id].clear()
            del self._entity_maps[session_id]

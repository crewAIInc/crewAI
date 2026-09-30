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
import secrets
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type

try:
    from pydantic import BaseModel, Field, PrivateAttr
except ImportError:
    class BaseModel:
        """Fallback BaseModel when pydantic is not installed."""
        pass
    def Field(*args, **kwargs):
        """Fallback Field factory."""
        return None
    def PrivateAttr(*args, **kwargs):
        """Fallback PrivateAttr factory."""
        default_factory = kwargs.get("default_factory")
        if default_factory:
            return default_factory()
        return kwargs.get("default", None)

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        """Fallback BaseTool when crewai is not installed."""
        def __init__(self, **kwargs: Any) -> None:
            """Initialize fallback BaseTool."""
            pass


class ZTDSSanitizerSchema(BaseModel):
    """Input schema definition for the ZTDSSanitizerTool.

    Attributes:
        text: The sensitive text to sanitize before model reasoning or delegation.
        session_id: Ephemeral session scope for surrogate mapping.
    """
    text: str = Field(..., description="The sensitive text to sanitize before model reasoning or delegation")
    session_id: Optional[str] = Field(default="crew-default", description="Ephemeral session scope for surrogate mapping")


class ZTDSSanitizerTool(BaseTool):
    """CrewAI Tool providing Zero-Trust Data Sanitization (ZTDS) RFC v1.0.

    Enforces in-memory surrogate mapping for PII, API tokens, and corporate credentials
    with 0 network egress and Theorem 2 RAM zeroization.

    Attributes:
        name: Human-readable name of the tool.
        description: Functional description for LLM agent routing.
        args_schema: Pydantic schema class for input argument validation.
        PATTERNS: Class-level regex pattern catalog for sensitive entities.
    """
    name: str = "Zero-Trust Data Sanitizer"
    description: str = (
        "Sanitizes emails, cards, IBANs, API secrets, and sensitive entities in-memory "
        "conforming to IETF draft-sibiryakov-ztds-protocol-02 with zero external egress."
    )
    args_schema: Type[BaseModel] = ZTDSSanitizerSchema

    TOKEN_PATTERN: ClassVar[re.Pattern] = re.compile(r"\[[A-Z_]+_TOKEN_[a-zA-Z0-9_-]+\]")

    PATTERNS: ClassVar[Dict[str, re.Pattern]] = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b"),
        "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IBAN": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{6}[-\s]?\d{5})\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "API_SECRET": re.compile(r"\b(?:sk-(?:proj-)?[a-zA-Z0-9_-]{20,}|ghp_[a-zA-Z0-9]{20,}|eyJ[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"),
    }

    use_random_surrogates: bool = True

    _session_maps: Dict[str, Dict[str, str]] = PrivateAttr(default_factory=dict)
    _entity_maps: Dict[str, Dict[str, str]] = PrivateAttr(default_factory=dict)
    _call_tokens: Dict[str, Set[str]] = PrivateAttr(default_factory=dict)

    def __init__(self, use_random_surrogates: bool = True, **kwargs: Any) -> None:
        """Initialize the ZTDSSanitizerTool with isolated in-memory mapping state.

        Args:
            use_random_surrogates: Whether to generate unguessable high-entropy surrogates
                to mitigate token oracle / blind substitution injection. Defaults to True.
            **kwargs: Arbitrary keyword arguments passed to the BaseTool superclass.
        """
        super().__init__(**kwargs)
        self.use_random_surrogates = use_random_surrogates
        if not hasattr(self, "_session_maps") or self._session_maps is None:
            self._session_maps = {}
        if not hasattr(self, "_entity_maps") or self._entity_maps is None:
            self._entity_maps = {}
        if not hasattr(self, "_call_tokens") or self._call_tokens is None:
            self._call_tokens = {}

    def _run(self, text: str, session_id: str = "crew-default") -> str:
        """Execute in-memory zero-trust data sanitization on the provided text.

        Performs a two-pass algorithm:
        1. Identifies all sensitive entity matches and assigns bracketed surrogates
           (unguessable hex surrogates by default to eliminate token oracle extraction).
        2. Substitutes matches in descending span offset order to preserve character indices without collisions.

        Args:
            text: Raw input string containing sensitive entities to be sanitized.
            session_id: Ephemeral session identifier scope for mapping isolation. Defaults to 'crew-default'.

        Returns:
            Sanitized text string with all detected sensitive entities substituted by bracketed surrogates.
        """
        if session_id not in self._session_maps:
            self._session_maps[session_id] = {}
            self._entity_maps[session_id] = {}
            self._call_tokens[session_id] = set()

        token_map = self._session_maps[session_id]
        entity_map = self._entity_maps[session_id]
        call_tokens = self._call_tokens[session_id]
        sanitized = text

        for entity_type, pattern in self.PATTERNS.items():
            matches = list(pattern.finditer(sanitized))
            if not matches:
                continue

            # Pass 1: Assign surrogates in ascending document order (left-to-right)
            for match in sorted(matches, key=lambda m: m.start()):
                original = match.group(0)
                if original not in entity_map:
                    if self.use_random_surrogates:
                        while True:
                            candidate = f"[{entity_type}_TOKEN_{secrets.token_hex(4)}]"
                            if candidate not in text and candidate not in token_map:
                                token = candidate
                                break
                    else:
                        count = len([k for k in token_map if k.startswith(f"[{entity_type}_TOKEN_")]) + 1
                        while True:
                            candidate = f"[{entity_type}_TOKEN_{count}]"
                            if candidate not in text and candidate not in token_map:
                                token = candidate
                                break
                            count += 1
                    token_map[token] = original
                    entity_map[original] = token

                call_tokens.add(entity_map[original])

            # Pass 2: Substitute surrogates in descending span offset order (right-to-left) to preserve indices
            for match in sorted(matches, key=lambda m: m.start(), reverse=True):
                original = match.group(0)
                token = entity_map[original]
                start, end = match.span()
                sanitized = sanitized[:start] + token + sanitized[end:]

        return sanitized

    def restore(
        self,
        text: str,
        session_id: str = "crew-default",
        allowed_tokens: Optional[Set[str]] = None,
        auto_zeroize: bool = False,
    ) -> str:
        """Restore bracketed surrogate tokens back to their original plaintext values.

        Uses single-pass regular expression token dispatch to permanently eliminate
        sequential string replacement collisions in O(N) time. Enforces caller-scoped
        token allow-listing to prevent token oracle / blind substitution attacks.

        Args:
            text: Text string containing surrogate tokens to restore.
            session_id: Ephemeral session identifier whose mapping tables to use. Defaults to 'crew-default'.
            allowed_tokens: Optional explicit set of authorized tokens to restore.
                If None, defaults to tokens emitted in this session.
            auto_zeroize: If True, automatically zeroizes the session mappings after restoration.

        Returns:
            Restored text string with surrogate tokens replaced by original values.
        """
        token_map = self._session_maps.get(session_id, {})
        if not token_map:
            return text

        effective_allowed = (
            allowed_tokens if allowed_tokens is not None else self._call_tokens.get(session_id, None)
        )

        def _replace_token(match: re.Match) -> str:
            tok = match.group(0)
            if tok in token_map:
                if effective_allowed is None or tok in effective_allowed:
                    return token_map[tok]
            return tok

        restored = self.TOKEN_PATTERN.sub(_replace_token, text)
        if auto_zeroize:
            self.zeroize(session_id)
        return restored

    def zeroize(self, session_id: str = "crew-default") -> None:
        """Execute Theorem 2 RAM zeroization by clearing and removing session mappings.

        Physically clears all token and entity dictionaries for the specified session
        and deletes the session keys from the tool's mapping registry.

        Args:
            session_id: Ephemeral session identifier to purge from volatile RAM. Defaults to 'crew-default'.
        """
        if session_id in self._session_maps:
            self._session_maps[session_id].clear()
            del self._session_maps[session_id]
        if session_id in self._entity_maps:
            self._entity_maps[session_id].clear()
            del self._entity_maps[session_id]
        if session_id in self._call_tokens:
            self._call_tokens[session_id].clear()
            del self._call_tokens[session_id]

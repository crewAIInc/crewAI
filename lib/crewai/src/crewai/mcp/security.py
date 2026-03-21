"""MCP security layer for CrewAI agents.

This module provides cryptographic identity, message signing, tool integrity
verification, and replay protection for MCP tool calls, addressing the security
gaps described in OWASP MCP Top 10.

It wraps the ``mcp-secure`` library (optional dependency) and exposes:

- :class:`MCPSecurityConfig` -- Pydantic model for security settings.
- :class:`MCPSecurityManager` -- Stateful manager handling key generation,
  passport creation/signing, message signing/verification, tool integrity
  checks, and nonce-based replay protection.

When ``mcp-secure`` is not installed, :meth:`MCPSecurityManager.is_available`
returns ``False`` and all operations gracefully degrade to no-ops.

Example:
    ```python
    from crewai.mcp.security import MCPSecurityConfig, MCPSecurityManager

    config = MCPSecurityConfig(
        agent_name="my-agent",
        agent_version="1.0.0",
        capabilities=["read", "write"],
    )
    manager = MCPSecurityManager(config)

    # Sign a tool call message
    message = {"jsonrpc": "2.0", "method": "tools/call", "id": 1}
    envelope = manager.sign_message(message)

    # Verify a tool definition
    tool = {"name": "read_file", "description": "...", "inputSchema": {}}
    signature = "..."  # from server
    is_valid = manager.verify_tool(tool, signature)
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


def _mcp_secure_available() -> bool:
    """Check whether the ``mcp-secure`` package is importable."""
    try:
        import mcp_secure  # noqa: F401

        return True
    except ImportError:
        return False


class MCPSecurityConfig(BaseModel):
    """Configuration for MCP security features.

    All fields have sensible defaults so that a minimal configuration like
    ``MCPSecurityConfig()`` auto-generates keys and creates an unsigned
    passport. For production use, supply a Trust Authority key pair so that
    passports are cryptographically signed and verifiable.

    Example:
        ```python
        # Minimal -- keys auto-generated, passport self-signed
        config = MCPSecurityConfig(agent_name="researcher")

        # Production -- TA-signed passport, tool verification enabled
        config = MCPSecurityConfig(
            agent_name="researcher",
            agent_version="2.0.0",
            capabilities=["search", "summarize"],
            ta_public_key="<PEM public key>",
            ta_private_key="<PEM private key>",
            verify_tool_signatures=True,
            sign_messages=True,
        )
        ```
    """

    agent_name: str = Field(
        default="crewai-agent",
        description="Human-readable name for the agent passport.",
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Semantic version included in the agent passport.",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of capability strings for the agent passport.",
    )

    # Agent key pair -- auto-generated when left as ``None``.
    private_key: str | None = Field(
        default=None,
        description=(
            "PEM-encoded ECDSA P-256 private key for signing messages. "
            "Auto-generated if not provided."
        ),
    )
    public_key: str | None = Field(
        default=None,
        description=(
            "PEM-encoded ECDSA P-256 public key matching the private key. "
            "Auto-generated if not provided."
        ),
    )

    # Trust Authority key pair -- used to sign and verify passports.
    ta_private_key: str | None = Field(
        default=None,
        description=(
            "PEM-encoded private key of the Trust Authority. "
            "When provided, passports are TA-signed."
        ),
    )
    ta_public_key: str | None = Field(
        default=None,
        description=(
            "PEM-encoded public key of the Trust Authority. "
            "When provided, incoming passports can be verified."
        ),
    )

    sign_messages: bool = Field(
        default=True,
        description="Whether to sign outgoing MCP JSON-RPC messages.",
    )
    verify_tool_signatures: bool = Field(
        default=True,
        description="Whether to verify tool definition signatures from the server.",
    )


class MCPSecurityManager:
    """Manages MCP security operations for a single agent.

    Handles key generation, passport lifecycle, message signing/verification,
    tool integrity checks, and replay protection via nonce tracking.

    The manager is safe to share across MCP clients that belong to the same
    agent; each call to :meth:`sign_message` generates a unique nonce.
    """

    def __init__(self, config: MCPSecurityConfig) -> None:
        self._config = config
        self._passport: dict[str, Any] | None = None
        self._nonce_store: Any | None = None
        self._private_key: str | None = config.private_key
        self._public_key: str | None = config.public_key

        if _mcp_secure_available():
            self._initialize()

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` when the ``mcp-secure`` package is installed."""
        return _mcp_secure_available()

    @property
    def passport(self) -> dict[str, Any] | None:
        """Return the current agent passport, or ``None`` if unavailable."""
        return self._passport

    @property
    def passport_id(self) -> str | None:
        """Return the passport ID, or ``None`` if no passport exists."""
        if self._passport is None:
            return None
        return self._passport.get("passport_id")

    @property
    def public_key(self) -> str | None:
        """Return the agent's public key."""
        return self._public_key

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Generate keys (if needed), create passport, and set up nonce store."""
        from mcp_secure import (
            NonceStore,
            create_passport,
            generate_key_pair,
            sign_passport,
        )

        # Generate agent keys if not provided
        if self._private_key is None or self._public_key is None:
            keys = generate_key_pair()
            self._private_key = keys["private_key"]
            self._public_key = keys["public_key"]

        # Create passport
        passport = create_passport(
            name=self._config.agent_name,
            version=self._config.agent_version,
            public_key=self._public_key,
            capabilities=self._config.capabilities,
        )

        # Sign with Trust Authority key if available
        if self._config.ta_private_key is not None:
            passport = sign_passport(passport, self._config.ta_private_key)

        self._passport = passport
        self._nonce_store = NonceStore()

    # ------------------------------------------------------------------
    # Message signing / verification
    # ------------------------------------------------------------------

    def sign_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Sign an outgoing MCP JSON-RPC message.

        Returns the original *message* unchanged when security is unavailable
        or message signing is disabled.

        Args:
            message: The JSON-RPC message dict to sign.

        Returns:
            A signed envelope dict (when signing is active) or the original
            message (when signing is inactive).
        """
        if not self.is_available() or not self._config.sign_messages:
            return message

        if self._passport is None or self._private_key is None:
            return message

        from mcp_secure import sign_message

        return sign_message(message, self._passport["passport_id"], self._private_key)

    def verify_message(self, envelope: dict[str, Any], sender_public_key: str) -> bool:
        """Verify an incoming signed MCP message envelope.

        Args:
            envelope: The signed envelope dict.
            sender_public_key: PEM-encoded public key of the sender.

        Returns:
            ``True`` if the envelope signature and nonce are valid.
        """
        if not self.is_available():
            return True

        from mcp_secure import verify_message

        result = verify_message(envelope, sender_public_key)
        if not result.get("valid", False):
            logger.warning("MCP message verification failed: %s", result.get("error"))
            return False

        # Replay protection
        nonce = envelope.get("nonce")
        if nonce is not None and self._nonce_store is not None:
            if not self._nonce_store.check(nonce):
                logger.warning("MCP message replay detected (duplicate nonce)")
                return False

        return True

    # ------------------------------------------------------------------
    # Tool integrity
    # ------------------------------------------------------------------

    def sign_tool(self, tool_definition: dict[str, Any]) -> str | None:
        """Sign a tool definition.

        Args:
            tool_definition: Tool definition dict (name, description, inputSchema).

        Returns:
            Signature string, or ``None`` when security is unavailable.
        """
        if not self.is_available() or self._private_key is None:
            return None

        from mcp_secure import sign_tool

        return sign_tool(tool_definition, self._private_key)

    def verify_tool(
        self,
        tool_definition: dict[str, Any],
        signature: str,
        signer_public_key: str | None = None,
    ) -> bool:
        """Verify a tool definition signature.

        Args:
            tool_definition: Tool definition dict.
            signature: Signature string to verify.
            signer_public_key: PEM-encoded public key of the signer.
                Falls back to the agent's own public key if ``None``.

        Returns:
            ``True`` if the signature is valid, ``False`` otherwise.
            Always returns ``True`` when security is unavailable or
            tool verification is disabled.
        """
        if not self.is_available() or not self._config.verify_tool_signatures:
            return True

        key = signer_public_key or self._public_key
        if key is None:
            return True

        from mcp_secure import verify_tool

        return verify_tool(tool_definition, signature, key)

    # ------------------------------------------------------------------
    # Passport verification
    # ------------------------------------------------------------------

    def verify_passport(
        self, passport: dict[str, Any], ta_public_key: str | None = None
    ) -> bool:
        """Verify a remote passport signature.

        Args:
            passport: Passport dict to verify.
            ta_public_key: Trust Authority public key. Falls back to the
                configured ``ta_public_key`` if ``None``.

        Returns:
            ``True`` if the passport signature is valid.
        """
        if not self.is_available():
            return True

        key = ta_public_key or self._config.ta_public_key
        if key is None:
            return True

        from mcp_secure import verify_passport_signature

        return verify_passport_signature(passport, key)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Release resources held by the security manager."""
        if self._nonce_store is not None:
            self._nonce_store.destroy()
            self._nonce_store = None
        self._passport = None

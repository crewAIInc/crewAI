"""
SwiftAPI Attestation Client for CrewAI

Handles cryptographic attestation verification for tool invocations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .config import SwiftAPIConfig

logger = logging.getLogger(__name__)


class AttestationError(Exception):
    """Base exception for attestation failures."""

    pass


class PolicyViolationError(AttestationError):
    """Raised when an action is denied by SwiftAPI policy."""

    def __init__(self, message: str, denial_reason: Optional[str] = None):
        super().__init__(message)
        self.denial_reason = denial_reason or message


@dataclass
class AttestationResult:
    """Result of an attestation request."""

    approved: bool
    jti: Optional[str] = None  # JWT Token ID for audit trail
    signature: Optional[str] = None  # Ed25519 signature
    reason: Optional[str] = None
    expires_at: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def denied(cls, reason: str) -> AttestationResult:
        return cls(approved=False, reason=reason)

    @classmethod
    def approved_with_jti(cls, jti: str, signature: Optional[str] = None) -> AttestationResult:
        return cls(approved=True, jti=jti, signature=signature)


class AttestationProvider(ABC):
    """Abstract base for attestation providers."""

    @abstractmethod
    async def verify_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AttestationResult:
        """Verify an action and return attestation result."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class SwiftAPIAttestationProvider(AttestationProvider):
    """SwiftAPI attestation provider using the /verify endpoint."""

    def __init__(self, config: SwiftAPIConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "X-SwiftAPI-Authority": self.config.api_key or "",
                    "Content-Type": "application/json",
                    "User-Agent": "CrewAI-SwiftAPI/1.0",
                },
            )
        return self._client

    def _generate_fingerprint(self, action_type: str, params: Dict[str, Any]) -> str:
        """Generate deterministic fingerprint for the action."""
        data = {"action": action_type, "params": params}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    async def verify_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AttestationResult:
        """Verify an action against SwiftAPI policies.

        Args:
            action_type: Type of action (e.g., 'tool_invocation', 'agent_handoff')
            action_params: Parameters for the action
            intent: Human-readable description of what the action does
            context: Additional context (agent name, crew name, etc.)

        Returns:
            AttestationResult with approval status and JTI if approved.

        Raises:
            PolicyViolationError: If action is explicitly denied.
            AttestationError: If attestation request fails.
        """
        if not self.config.is_configured:
            raise AttestationError(
                "SwiftAPI not configured. Set SWIFTAPI_KEY environment variable."
            )

        # Build request payload matching SwiftAPI expected format
        request_id = f"crewai_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        payload = {
            "action": {
                "type": action_type,
                "intent": intent,
                "params": action_params,
            },
            "context": {
                "app_id": self.config.app_id,
                "actor": context.get("agent_name", self.config.actor) if context else self.config.actor,
                "environment": "production",
                "request_id": request_id,
                **(context or {}),
            },
        }

        try:
            client = await self._get_client()
            response = await client.post("/verify", json=payload)

            if response.status_code == 200:
                data = response.json()
                # Handle both old and new response formats
                jti = data.get("jti") or data.get("verification_id") or data.get("attestation", {}).get("jti")
                return AttestationResult(
                    approved=True,
                    jti=jti,
                    signature=data.get("signature") or data.get("attestation", {}).get("signature"),
                    expires_at=data.get("expires_at"),
                    raw_response=data,
                )

            elif response.status_code == 403:
                data = response.json()
                detail = data.get("detail", {})
                if isinstance(detail, dict):
                    reason = detail.get("message") or detail.get("reason") or "Policy denied this action"
                else:
                    reason = str(detail) or data.get("reason", "Policy denied this action")
                raise PolicyViolationError(
                    f"Action denied by SwiftAPI policy: {reason}",
                    denial_reason=reason,
                )

            elif response.status_code == 401:
                raise AttestationError(
                    "SwiftAPI authentication failed. Check your API key."
                )

            elif response.status_code == 429:
                raise AttestationError(
                    "SwiftAPI rate limit exceeded. Too many requests."
                )

            else:
                error_text = response.text[:200] if response.text else "Unknown error"
                raise AttestationError(
                    f"SwiftAPI returned status {response.status_code}: {error_text}"
                )

        except httpx.TimeoutException:
            if self.config.fail_open:
                logger.warning(
                    "[SwiftAPI] Timeout - fail_open=True, allowing action (DANGEROUS)"
                )
                return AttestationResult(approved=True, reason="fail_open timeout bypass")
            raise AttestationError(
                f"SwiftAPI request timed out after {self.config.timeout}s. "
                "Action blocked (fail-closed)."
            )

        except httpx.ConnectError as e:
            if self.config.fail_open:
                logger.warning(
                    "[SwiftAPI] Connection failed - fail_open=True, allowing action (DANGEROUS)"
                )
                return AttestationResult(approved=True, reason="fail_open connection bypass")
            raise AttestationError(
                f"Cannot connect to SwiftAPI at {self.config.base_url}: {e}. "
                "Action blocked (fail-closed)."
            )

        except (PolicyViolationError, AttestationError):
            raise

        except Exception as e:
            if self.config.fail_open:
                logger.warning(
                    f"[SwiftAPI] Unexpected error - fail_open=True, allowing action: {e}"
                )
                return AttestationResult(approved=True, reason=f"fail_open error bypass: {e}")
            raise AttestationError(f"SwiftAPI attestation failed: {e}")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class MockAttestationProvider(AttestationProvider):
    """Mock provider for testing. Approves everything."""

    def __init__(self, approve_all: bool = True):
        self.approve_all = approve_all
        self.call_log: List[Dict[str, Any]] = []

    async def verify_action(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        intent: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AttestationResult:
        self.call_log.append({
            "action_type": action_type,
            "action_params": action_params,
            "intent": intent,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if self.approve_all:
            return AttestationResult(
                approved=True,
                jti=f"mock-jti-{len(self.call_log)}",
                reason="mock approval",
            )
        else:
            return AttestationResult.denied("mock denial for testing")

    async def close(self) -> None:
        pass

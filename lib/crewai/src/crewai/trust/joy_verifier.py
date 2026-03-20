"""Joy trust network verification for CrewAI agent delegation.

This module integrates CrewAI with Joy, an open trust network for AI agents.
It enables crews to verify agent trustworthiness before delegation.

Usage:
    from crewai.trust import JoyVerifier

    verifier = JoyVerifier(min_trust_score=0.5)
    if verifier.should_trust("ag_xxx"):
        # Safe to delegate
        pass

Environment Variables:
    JOY_API_URL: Joy API endpoint (default: https://joy-connect.fly.dev)
    JOY_API_KEY: Your Joy API key (optional, for authenticated operations)
    JOY_AGENT_ID: Your agent's Joy identity (optional)
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

# Joy agent IDs follow pattern: ag_ followed by 24 hex characters
AGENT_ID_PATTERN = re.compile(r"^ag_[a-f0-9]{24}$")


def _validate_agent_id(agent_id: str) -> str:
    """Validate and sanitize agent ID to prevent injection attacks.

    Args:
        agent_id: The agent ID to validate

    Returns:
        The validated agent ID

    Raises:
        ValueError: If the agent ID format is invalid
    """
    if not agent_id or not isinstance(agent_id, str):
        raise ValueError("Agent ID must be a non-empty string")

    agent_id = agent_id.strip().lower()

    if not AGENT_ID_PATTERN.match(agent_id):
        raise ValueError(
            f"Invalid agent ID format: {agent_id!r}. "
            "Expected format: ag_ followed by 24 hex characters"
        )

    return agent_id


class TrustVerificationError(Exception):
    """Raised when trust verification fails."""
    pass


@dataclass
class VerificationResult:
    """Result of agent trust verification."""
    is_trusted: bool
    agent_id: str
    trust_score: float
    vouch_count: int
    verified: bool
    capabilities: List[str]
    error: Optional[str] = None

    def meets_threshold(self, min_score: float) -> bool:
        """Check if trust score meets minimum threshold."""
        return self.trust_score >= min_score


class JoyVerifier:
    """Verify agent trust using Joy network.

    Joy is a decentralized trust network where agents vouch for each other.
    This verifier checks an agent's trust score before allowing delegation.

    Example:
        verifier = JoyVerifier(min_trust_score=0.5)

        # Verify an agent
        result = verifier.verify_agent("ag_xxx")
        if result.is_trusted:
            # Safe to delegate
            pass

        # Use as middleware
        if verifier.should_trust("ag_xxx"):
            delegate_task(agent)
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_trust_score: float = 0.5,
        require_verified: bool = False,
    ):
        """Initialize Joy verifier.

        Args:
            api_url: Joy API URL (default: from JOY_API_URL env or https://joy-connect.fly.dev)
            api_key: Joy API key (default: from JOY_API_KEY env)
            agent_id: Your agent's Joy ID (default: from JOY_AGENT_ID env)
            min_trust_score: Minimum trust score to consider agent trusted (0.0-2.0)
            require_verified: If True, only trust agents with verified badge
        """
        self.api_url = (
            api_url
            or os.getenv("JOY_API_URL")
            or "https://joy-connect.fly.dev"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("JOY_API_KEY")
        self.agent_id = agent_id or os.getenv("JOY_AGENT_ID")
        self.min_trust_score = min_trust_score
        self.require_verified = require_verified
        self._client = None

    def _get_client(self):
        """Lazy load HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=30.0)
            except ImportError:
                raise ImportError(
                    "httpx is required for Joy integration. "
                    "Install with: pip install httpx"
                )
        return self._client

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make HTTP request to Joy API."""
        client = self._get_client()
        url = f"{self.api_url}{path}"

        headers = kwargs.pop("headers", {})
        headers["User-Agent"] = "crewai-joy/1.0.0"
        if self.api_key:
            headers["x-api-key"] = self.api_key

        response = client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

    def verify_agent(self, agent_id: str) -> VerificationResult:
        """Verify an agent's trust status.

        Args:
            agent_id: Joy agent ID to verify (e.g., "ag_xxx")

        Returns:
            VerificationResult with trust details

        Raises:
            TrustVerificationError: If verification fails
            ValueError: If agent_id format is invalid
        """
        # Validate agent ID before try block so validation errors propagate immediately
        validated_id = _validate_agent_id(agent_id)

        try:
            data = self._request("GET", f"/agents/{quote(validated_id, safe='')}")

            trust_score = float(data.get("trust_score", 0))
            vouch_count = int(data.get("vouch_count", 0))
            verified = bool(data.get("verified", False))
            capabilities = data.get("capabilities", [])

            # Determine if trusted based on criteria
            is_trusted = trust_score >= self.min_trust_score
            if self.require_verified and not verified:
                is_trusted = False

            result = VerificationResult(
                is_trusted=is_trusted,
                agent_id=validated_id,
                trust_score=trust_score,
                vouch_count=vouch_count,
                verified=verified,
                capabilities=capabilities,
            )

            logger.info(
                f"Agent {validated_id} verification: "
                f"score={trust_score}, vouches={vouch_count}, "
                f"verified={verified}, trusted={is_trusted}"
            )

            return result

        except ImportError:
            # Re-raise ImportError so users know to install dependencies
            raise
        except Exception as e:
            logger.error(f"Trust verification failed for {validated_id}: {e}")
            return VerificationResult(
                is_trusted=False,
                agent_id=validated_id,
                trust_score=0.0,
                vouch_count=0,
                verified=False,
                capabilities=[],
                error=str(e),
            )

    def should_trust(self, agent_id: str, raise_on_error: bool = False) -> bool:
        """Simple check if an agent should be trusted.

        Args:
            agent_id: Joy agent ID to check
            raise_on_error: If True, raise TrustVerificationError on API/network errors
                          instead of returning False

        Returns:
            True if agent meets trust criteria

        Raises:
            TrustVerificationError: If raise_on_error=True and verification fails due to
                                   API/network error (not due to low trust score)
        """
        result = self.verify_agent(agent_id)

        # Surface API errors if requested
        if result.error and raise_on_error:
            raise TrustVerificationError(
                f"Trust verification failed for {agent_id}: {result.error}"
            )

        return result.is_trusted

    def verify_before_delegation(
        self,
        agent_id: str,
        required_capabilities: Optional[List[str]] = None,
    ) -> bool:
        """Verify agent before delegating a task.

        Args:
            agent_id: Joy agent ID to verify
            required_capabilities: List of capabilities the agent must have

        Returns:
            True if agent is trusted and has required capabilities

        Raises:
            TrustVerificationError: If agent fails verification
        """
        result = self.verify_agent(agent_id)

        # Surface API errors instead of masking them as "not trusted"
        if result.error:
            raise TrustVerificationError(
                f"Trust verification failed for {agent_id}: {result.error}"
            )

        if not result.is_trusted:
            raise TrustVerificationError(
                f"Agent {agent_id} not trusted: "
                f"score={result.trust_score} (min={self.min_trust_score})"
            )

        if required_capabilities:
            missing = set(required_capabilities) - set(result.capabilities)
            if missing:
                raise TrustVerificationError(
                    f"Agent {agent_id} missing capabilities: {missing}"
                )

        return True

    def discover_trusted_agents(
        self,
        capability: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 10,
        raise_on_error: bool = False,
    ) -> List[VerificationResult]:
        """Discover trusted agents from Joy network.

        Args:
            capability: Filter by capability (e.g., "code", "research")
            query: Search query
            limit: Maximum results
            raise_on_error: If True, raise TrustVerificationError on API/network errors
                          instead of returning empty list

        Returns:
            List of VerificationResult for trusted agents

        Raises:
            TrustVerificationError: If raise_on_error=True and discovery fails
        """
        params = {"limit": limit}
        if capability:
            params["capability"] = capability
        if query:
            params["query"] = query

        try:
            data = self._request("GET", "/agents/discover", params=params)
            agents = data.get("agents", [])

            results = []
            for agent in agents:
                trust_score = float(agent.get("trust_score", 0))
                verified = bool(agent.get("verified", False))

                # Apply same trust criteria as verify_agent
                is_trusted = trust_score >= self.min_trust_score
                if self.require_verified and not verified:
                    is_trusted = False

                result = VerificationResult(
                    is_trusted=is_trusted,
                    agent_id=agent.get("id", ""),
                    trust_score=trust_score,
                    vouch_count=int(agent.get("vouch_count", 0)),
                    verified=verified,
                    capabilities=agent.get("capabilities", []),
                )
                if result.is_trusted:
                    results.append(result)

            return results

        except ImportError:
            # Re-raise ImportError so users know to install dependencies
            raise
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
            if raise_on_error:
                raise TrustVerificationError(f"Agent discovery failed: {e}")
            return []

    def get_vouch_suggestions(self, limit: int = 10) -> dict:
        """Get suggestions for agents to vouch for.

        Requires agent_id to be set.

        Returns:
            Dict with suggestion categories: similar, voucherNetwork, complementary
        """
        if not self.agent_id:
            raise ValueError("agent_id required for vouch suggestions")

        # Use params dict for proper URL encoding (prevents injection)
        return self._request(
            "GET",
            "/agents/vouch-suggestions",
            params={"agentId": self.agent_id, "limit": limit}
        )

    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

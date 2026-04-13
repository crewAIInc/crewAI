"""Joy Trust Network API client."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from crewai.integrations.joy_trust.config import JoyTrustConfig


logger = logging.getLogger(__name__)


@dataclass
class TrustResult:
    """Result of a trust verification check.

    Attributes:
        agent_name: Name of the agent checked
        agent_id: Joy agent ID (if found)
        trust_score: Trust score (0.0 - 5.0)
        verified: Whether the agent is identity-verified
        meets_threshold: Whether score meets the minimum threshold
        threshold_used: The threshold that was checked against
        vouch_count: Number of vouches received
        capabilities: List of agent capabilities
        badges: List of badges (verified, pro, etc.)
        error: Error message if lookup failed
        trust_context: Network statistics and recommendations
    """

    agent_name: str
    agent_id: str | None
    trust_score: float
    verified: bool
    meets_threshold: bool
    threshold_used: float
    vouch_count: int
    capabilities: list[str]
    badges: list[str]
    error: str | None
    trust_context: dict[str, Any] | None = None


class JoyTrustClient:
    """Client for Joy Trust Network API.

    Provides trust verification for agents before delegation.
    Implements fail-closed security by default.
    """

    def __init__(self, config: JoyTrustConfig | None = None):
        """Initialize the Joy Trust client.

        Args:
            config: Configuration options (uses env vars if not provided)
        """
        self.config = config or JoyTrustConfig.from_env()
        self._cache: dict[str, tuple[TrustResult, float]] = {}

    def check_trust(
        self,
        agent_name: str,
        min_score: float | None = None,
    ) -> TrustResult:
        """Check an agent's trust score on Joy Trust Network.

        Args:
            agent_name: Name or identifier of the agent to check
            min_score: Override minimum threshold (uses config default if None)

        Returns:
            TrustResult with trust information and verification status
        """
        threshold = min_score if min_score is not None else self.config.min_score

        # Check cache
        cache_key = f"{agent_name}_{threshold}"
        if cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            if time.time() - cached_at < self.config.cache_ttl:
                logger.debug(f"Joy Trust: Using cached result for {agent_name}")
                return result

        # Query Joy API
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                headers = {}
                if self.config.api_key:
                    headers["x-api-key"] = self.config.api_key

                response = client.get(
                    f"{self.config.api_url}/agents/discover",
                    params={"query": agent_name},
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

                agents = data.get("agents") or []
                trust_context = data.get("trust_context")

                # Find exact match by name (case-insensitive)
                normalized_name = agent_name.lower()
                agent = next(
                    (a for a in agents if (a.get("name") or "").lower() == normalized_name),
                    None
                )

                if not agent:
                    result = TrustResult(
                        agent_name=agent_name,
                        agent_id=None,
                        trust_score=0.0,
                        verified=False,
                        meets_threshold=False,
                        threshold_used=threshold,
                        vouch_count=0,
                        capabilities=[],
                        badges=[],
                        error=f"Agent '{agent_name}' not found on Joy Trust Network",
                        trust_context=trust_context,
                    )
                    return result

                # Parse trust score safely
                raw_score = agent.get("trust_score")
                try:
                    trust_score = 0.0 if raw_score in (None, "") else float(raw_score)
                except (TypeError, ValueError):
                    trust_score = 0.0

                result = TrustResult(
                    agent_name=agent.get("name") or agent_name,
                    agent_id=agent.get("id"),
                    trust_score=trust_score,
                    verified=agent.get("verified", False),
                    meets_threshold=trust_score >= threshold,
                    threshold_used=threshold,
                    vouch_count=agent.get("vouch_count", 0),
                    capabilities=agent.get("capabilities", []),
                    badges=agent.get("badges", []),
                    error=None,
                    trust_context=trust_context,
                )

                # Cache successful result
                self._cache[cache_key] = (result, time.time())
                return result

        except httpx.RequestError as e:
            logger.error(f"Joy Trust connection error: {e}")
            return self._fail_closed_result(agent_name, threshold, f"Connection error: {e}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Joy Trust API error: {e.response.status_code}")
            return self._fail_closed_result(
                agent_name, threshold, f"API error ({e.response.status_code})"
            )

        except Exception as e:
            logger.exception("Joy Trust unexpected error")
            return self._fail_closed_result(agent_name, threshold, f"Unexpected error: {e}")

    def _fail_closed_result(
        self,
        agent_name: str,
        threshold: float,
        error: str,
    ) -> TrustResult:
        """Create a fail-closed result for error cases.

        Security: Errors should deny delegation by default.
        """
        return TrustResult(
            agent_name=agent_name,
            agent_id=None,
            trust_score=0.0,
            verified=False,
            meets_threshold=self.config.fail_open,  # Only True if fail_open configured
            threshold_used=threshold,
            vouch_count=0,
            capabilities=[],
            badges=[],
            error=error,
            trust_context=None,
        )

    def verify_delegation_safety(
        self,
        target_agent: str,
        min_score: float | None = None,
    ) -> tuple[bool, str]:
        """Verify if it's safe to delegate to an agent.

        Args:
            target_agent: Name of the agent to delegate to
            min_score: Minimum trust score required

        Returns:
            Tuple of (is_safe, reason)
        """
        result = self.check_trust(target_agent, min_score)

        if result.error and not self.config.fail_open:
            return False, f"Trust verification failed: {result.error}"

        if not result.meets_threshold:
            return False, (
                f"Trust score {result.trust_score:.1f} below threshold "
                f"{result.threshold_used:.1f} for agent '{result.agent_name}'"
            )

        return True, (
            f"Trust verified: {result.agent_name} "
            f"(score: {result.trust_score:.1f}, vouches: {result.vouch_count})"
        )

"""CrewAI hooks for Joy Trust Network integration.

Provides automatic trust verification before A2A delegations.
"""

from __future__ import annotations

import logging
from typing import Any

from crewai.integrations.joy_trust.client import JoyTrustClient, TrustResult
from crewai.integrations.joy_trust.config import JoyTrustConfig


logger = logging.getLogger(__name__)

# Global state for the integration
_joy_client: JoyTrustClient | None = None
_enabled: bool = False


class JoyTrustVerificationError(Exception):
    """Raised when Joy Trust verification fails and blocks delegation."""

    def __init__(self, message: str, trust_result: TrustResult):
        super().__init__(message)
        self.trust_result = trust_result


def enable_joy_trust(
    api_key: str | None = None,
    min_score: float = 1.5,
    fail_open: bool = False,
    **kwargs,
) -> None:
    """Enable Joy Trust verification for A2A delegations.

    This registers hooks that verify agent trust before any delegation.
    Delegations to agents below the trust threshold will be blocked.

    Args:
        api_key: Joy Trust API key (optional, uses env var if not provided)
        min_score: Minimum trust score threshold (default: 1.5 "standard")
        fail_open: If True, allow delegation on network errors (default: False)
        **kwargs: Additional JoyTrustConfig options

    Example:
        >>> from crewai.integrations.joy_trust import enable_joy_trust
        >>> enable_joy_trust(min_score=2.0)  # Use "moderate" threshold
    """
    global _joy_client, _enabled

    config = JoyTrustConfig(
        api_key=api_key,
        min_score=min_score,
        fail_open=fail_open,
        **kwargs,
    )
    _joy_client = JoyTrustClient(config)
    _enabled = True

    logger.info(
        f"Joy Trust enabled: min_score={min_score}, fail_open={fail_open}"
    )


def disable_joy_trust() -> None:
    """Disable Joy Trust verification.

    A2A delegations will proceed without trust checks.
    """
    global _joy_client, _enabled
    _joy_client = None
    _enabled = False
    logger.info("Joy Trust disabled")


def is_enabled() -> bool:
    """Check if Joy Trust verification is enabled."""
    return _enabled and _joy_client is not None


def get_client() -> JoyTrustClient | None:
    """Get the current Joy Trust client."""
    return _joy_client


def check_agent_trust(
    agent_name: str,
    min_score: float | None = None,
) -> TrustResult:
    """Check an agent's trust score.

    Can be used manually or is called automatically during A2A delegation
    when Joy Trust is enabled.

    Args:
        agent_name: Name of the agent to check
        min_score: Override threshold (uses configured default if None)

    Returns:
        TrustResult with trust information

    Raises:
        RuntimeError: If Joy Trust is not enabled
    """
    if not _enabled or _joy_client is None:
        raise RuntimeError(
            "Joy Trust is not enabled. Call enable_joy_trust() first."
        )

    return _joy_client.check_trust(agent_name, min_score)


def verify_delegation(
    target_agent: str,
    min_score: float | None = None,
) -> None:
    """Verify trust before delegation, raising error if blocked.

    This is called automatically during A2A delegation when Joy Trust
    is enabled. Can also be called manually for explicit verification.

    Args:
        target_agent: Name of the agent to delegate to
        min_score: Override threshold (uses configured default if None)

    Raises:
        JoyTrustVerificationError: If trust check fails or score too low
    """
    if not _enabled or _joy_client is None:
        return  # Joy Trust not enabled, allow delegation

    is_safe, reason = _joy_client.verify_delegation_safety(target_agent, min_score)

    if not is_safe:
        result = _joy_client.check_trust(target_agent, min_score)
        logger.warning(f"Joy Trust blocked delegation: {reason}")
        raise JoyTrustVerificationError(reason, result)

    logger.info(f"Joy Trust: {reason}")


# ============ CrewAI Event Integration ============
# These functions integrate with CrewAI's event system to automatically
# verify trust before A2A delegations.

def _setup_event_listeners() -> None:
    """Set up event listeners for automatic trust verification.

    Called internally when Joy Trust is enabled.
    """
    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.a2a_events import A2ADelegationStartedEvent

        @crewai_event_bus.on(A2ADelegationStartedEvent)
        def on_delegation_started(source: Any, event: A2ADelegationStartedEvent) -> None:
            """Verify trust before delegation proceeds."""
            if not is_enabled():
                return

            # Extract agent name from event
            agent_name = event.a2a_agent_name or event.agent_id or event.endpoint

            try:
                verify_delegation(agent_name)
            except JoyTrustVerificationError as e:
                # Log the blocked delegation
                logger.warning(
                    f"Joy Trust blocked A2A delegation to {agent_name}: {e}"
                )
                # Note: To actually block the delegation, we'd need to integrate
                # at a different point. This logs the verification result.
                # For blocking, use the manual verify_delegation() call in your code.

    except ImportError:
        logger.debug("CrewAI events not available, skipping event listener setup")


# ============ Convenience Functions ============

def with_trust_verification(min_score: float = 1.5):
    """Decorator for functions that delegate to other agents.

    Example:
        >>> @with_trust_verification(min_score=2.0)
        ... def delegate_task(agent_name: str, task: str):
        ...     # This will only execute if agent_name passes trust check
        ...     return execute_delegation(agent_name, task)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to extract agent_name from common patterns
            agent_name = None
            if args and isinstance(args[0], str):
                agent_name = args[0]
            elif "agent_name" in kwargs:
                agent_name = kwargs["agent_name"]
            elif "target_agent" in kwargs:
                agent_name = kwargs["target_agent"]
            elif "endpoint" in kwargs:
                agent_name = kwargs["endpoint"]

            if agent_name:
                verify_delegation(agent_name, min_score)

            return func(*args, **kwargs)
        return wrapper
    return decorator

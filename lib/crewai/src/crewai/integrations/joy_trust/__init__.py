"""Joy Trust Network Integration for CrewAI.

Provides trust verification for agent-to-agent (A2A) delegations using the
Joy Trust Network - an independent, cross-platform trust layer for AI agents.

Features:
- Pre-delegation trust verification
- Configurable trust thresholds
- Fail-closed security (deny on error)
- Trust context with recommended thresholds

Usage:
    from crewai.integrations.joy_trust import enable_joy_trust

    # Enable trust verification for all A2A delegations
    enable_joy_trust(min_score=1.5)

    # Or with custom configuration
    enable_joy_trust(
        api_key="your_joy_api_key",
        min_score=2.0,  # moderate threshold
        fail_open=False,  # deny on network errors (default)
    )

Environment Variables:
    JOY_TRUST_API_KEY: Joy Trust API key (optional, increases rate limits)
    JOY_TRUST_MIN_SCORE: Minimum trust score threshold (default: 1.5)
    JOY_TRUST_ENABLED: Set to "true" to auto-enable (default: false)

For more information, see: https://choosejoy.com.au/docs/ENTERPRISE-ADOPTION.md
"""

from crewai.integrations.joy_trust.client import JoyTrustClient
from crewai.integrations.joy_trust.hooks import (
    enable_joy_trust,
    disable_joy_trust,
    check_agent_trust,
)
from crewai.integrations.joy_trust.config import JoyTrustConfig

__all__ = [
    "JoyTrustClient",
    "JoyTrustConfig",
    "enable_joy_trust",
    "disable_joy_trust",
    "check_agent_trust",
]

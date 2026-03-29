"""
AgentSIM tools for CrewAI — real carrier-grade phone number provisioning and OTP verification.

AgentSIM provides real T-Mobile SIM numbers that pass carrier lookup checks
as line_type: mobile. Unlike VoIP numbers (Twilio, Google Voice), these numbers
work with services that block virtual numbers (Google, Stripe, WhatsApp, etc.).

Install: pip install agentsim-sdk
API key: https://agentsim.dev/dashboard
Docs: https://agentsim.dev
"""

import asyncio
import os
from typing import Any, Optional

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


# ── Shared validation ──────────────────────────────────────────────


def _validate_agentsim_env() -> None:
    """Check agentsim-sdk is installed and API key is set. Called at tool init."""
    try:
        import agentsim  # noqa: F401
    except ImportError:
        raise ImportError(
            "`agentsim-sdk` package not found. Install with: pip install agentsim-sdk"
        ) from None

    if not os.getenv("AGENTSIM_API_KEY"):
        raise ValueError(
            "AGENTSIM_API_KEY environment variable is required. "
            "Get a key at https://agentsim.dev/dashboard"
        )


def _get_client():
    """Create a configured AgentSimClient."""
    from agentsim.client import AgentSimClient

    return AgentSimClient(
        api_key=os.environ["AGENTSIM_API_KEY"],
    )


def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Input schemas ──────────────────────────────────────────────────


class ProvisionInput(BaseModel):
    """Input schema for provisioning a phone number."""

    country: str = Field(
        default="US",
        description="ISO country code for the phone number (e.g. 'US', 'GB').",
    )
    agent_id: str = Field(
        default="crewai-agent",
        description="Identifier for the agent session, used for tracking.",
    )


class WaitForOtpInput(BaseModel):
    """Input schema for waiting for an OTP."""

    session_id: str = Field(
        description="The session_id returned by AgentSIMProvisionTool.",
    )
    timeout: int = Field(
        default=60,
        description="Maximum seconds to wait for the OTP to arrive.",
    )


class ReleaseInput(BaseModel):
    """Input schema for releasing a phone number."""

    session_id: str = Field(
        description="The session_id of the number to release.",
    )


# ── Tools ──────────────────────────────────────────────────────────


class AgentSIMProvisionTool(BaseTool):
    """Provision a real carrier-grade mobile phone number via AgentSIM.

    Returns a phone number in E.164 format and a session_id for OTP retrieval.
    The number is a real T-Mobile SIM that passes carrier lookup checks (not VoIP).
    Use this when a service requires SMS verification with a real mobile number.
    """

    name: str = "AgentSIM Provision Phone Number"
    description: str = (
        "Provision a real carrier-grade mobile phone number via AgentSIM. "
        "Returns a phone number in E.164 format (e.g. +14155551234) and a "
        "session_id needed for OTP retrieval. The number is a real T-Mobile "
        "SIM that passes carrier lookup checks — not VoIP. Use this when a "
        "service requires SMS verification with a real mobile number."
    )
    args_schema: type[BaseModel] = ProvisionInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["agentsim-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AGENTSIM_API_KEY",
                description="AgentSIM API key (get one at https://agentsim.dev/dashboard)",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _validate_agentsim_env()

    def _run(self, country: str = "US", agent_id: str = "crewai-agent") -> str:
        client = _get_client()
        try:
            result = _run_async(
                client.provision(agent_id=agent_id, country=country)
            )
            return (
                f"Phone number: {result.number} | "
                f"session_id: {result.session_id} | "
                f"carrier: T-Mobile | line_type: mobile"
            )
        finally:
            _run_async(client.aclose())


class AgentSIMWaitForOtpTool(BaseTool):
    """Wait for an SMS OTP to arrive on a provisioned AgentSIM number.

    Call this AFTER the phone number has been submitted to the target service.
    Returns the OTP code to enter in the verification field.
    """

    name: str = "AgentSIM Wait for OTP"
    description: str = (
        "Wait for an SMS OTP to arrive on a provisioned AgentSIM phone number. "
        "Call this AFTER the phone number form has been submitted to the target "
        "service. Provide the session_id from AgentSIM Provision Phone Number. "
        "Returns the OTP code to enter in the verification field."
    )
    args_schema: type[BaseModel] = WaitForOtpInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["agentsim-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AGENTSIM_API_KEY",
                description="AgentSIM API key",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _validate_agentsim_env()

    def _run(self, session_id: str, timeout: int = 60) -> str:
        from agentsim import OtpTimeoutError

        client = _get_client()
        try:
            otp = _run_async(
                client.wait_for_otp(session_id=session_id, timeout=timeout)
            )
            return f"OTP received: {otp.otp_code}"
        except OtpTimeoutError:
            return (
                "OTP timed out. Make sure the phone number was submitted to the "
                "target service before calling this tool. You can try again."
            )
        finally:
            _run_async(client.aclose())


class AgentSIMReleaseTool(BaseTool):
    """Release an AgentSIM phone number back to the pool.

    Always call this after SMS verification is complete to avoid extra charges.
    """

    name: str = "AgentSIM Release Phone Number"
    description: str = (
        "Release an AgentSIM phone number back to the pool after verification "
        "is complete. Always call this as the final cleanup step to avoid extra "
        "charges. Provide the session_id from provisioning."
    )
    args_schema: type[BaseModel] = ReleaseInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["agentsim-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AGENTSIM_API_KEY",
                description="AgentSIM API key",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _validate_agentsim_env()

    def _run(self, session_id: str) -> str:
        client = _get_client()
        try:
            _run_async(client.release(session_id=session_id))
            return f"Phone number released (session: {session_id})."
        finally:
            _run_async(client.aclose())

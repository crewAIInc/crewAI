"""
Spix tools for CrewAI.

Provides SpixCallTool, SpixSMSTool, and SpixEmailTool — drop-in CrewAI tools
that let any agent or crew make phone calls, send SMS, and send email via Spix.

Spix is communications infrastructure for AI agents: real phone numbers, voice
calls (~500ms latency), SMS, and email — all accessible via a simple REST API.

Install:
    pip install crewai-tools httpx

Usage:
    import os
    os.environ["SPIX_API_KEY"] = "your-api-key"

    from crewai_tools import SpixCallTool, SpixSMSTool, SpixEmailTool

    agent = Agent(
        role="Outreach Specialist",
        goal="Contact leads via the best channel",
        tools=[SpixCallTool(), SpixSMSTool(), SpixEmailTool()],
    )
"""

from __future__ import annotations

import os
from typing import Optional, Type

import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

SPIX_API_BASE = "https://api.spix.sh/v1"


def _get_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("SPIX_API_KEY")
    if not key:
        raise ValueError(
            "Spix API key not found. Pass api_key= or set the SPIX_API_KEY "
            "environment variable. Get your key at https://app.spix.sh/api-keys"
        )
    return key


def _spix_post(path: str, payload: dict, api_key: str) -> dict:
    """POST to Spix API and return parsed data. Raises on HTTP or API errors."""
    url = f"{SPIX_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=30)
        data = resp.json()
    except httpx.TimeoutException:
        raise RuntimeError(f"Spix API request timed out: POST {path}")
    except Exception as exc:
        raise RuntimeError(f"Spix API request failed: {exc}") from exc

    if not data.get("ok"):
        error = data.get("error", {})
        code = error.get("code", "unknown_error")
        message = error.get("message", "An unknown error occurred")
        raise RuntimeError(f"Spix API error [{code}]: {message}")

    return data["data"]


# ---------------------------------------------------------------------------
# SpixCallTool
# ---------------------------------------------------------------------------


class _CallInput(BaseModel):
    to: str = Field(
        description="The E.164 phone number to call, e.g. '+19175550123'."
    )
    playbook_id: str = Field(
        description=(
            "The Spix call playbook ID, e.g. 'cmp_call_abc123'. "
            "Defines the AI persona, script, and success criteria for the call."
        )
    )
    sender: str = Field(
        description=(
            "The E.164 Spix number to call from, e.g. '+14155550101'. "
            "Must be rented on your account and bound to this playbook."
        )
    )


class SpixCallTool(BaseTool):
    """Place an outbound AI phone call via Spix.

    The call runs a Spix playbook — an AI persona with a configured voice,
    script, and success criteria. Returns immediately with a session ID;
    the call runs asynchronously on Spix's voice engine (~500ms latency).

    Requires the ``SPIX_API_KEY`` environment variable or the ``api_key``
    constructor argument. Get a key at https://app.spix.sh/api-keys.

    Args:
        api_key: Spix API key. Falls back to the ``SPIX_API_KEY`` env var.

    Example:
        .. code-block:: python

            tool = SpixCallTool()

            agent = Agent(
                role="Sales Rep",
                goal="Confirm demo appointments by phone",
                tools=[tool],
            )
    """

    name: str = "Spix Call"
    description: str = (
        "Place an outbound AI phone call using Spix. "
        "The call runs a playbook that defines the AI persona, voice, and script. "
        "Use when you need to speak to a person by phone. "
        "Required inputs: to (E.164 phone number), playbook_id, "
        "sender (E.164 Spix number to call from)."
    )
    args_schema: Type[BaseModel] = _CallInput
    api_key: Optional[str] = None

    def _run(self, to: str, playbook_id: str, sender: str) -> str:
        key = _get_api_key(self.api_key)
        result = _spix_post(
            "/calls",
            {"to": to, "playbook_id": playbook_id, "sender": sender},
            key,
        )
        session_id = result.get("session_id", "unknown")
        status = result.get("status", "unknown")
        return (
            f"Call placed successfully. "
            f"Session ID: {session_id}. "
            f"Status: {status}. "
            f"Track live: spix watch transcript {session_id}"
        )


# ---------------------------------------------------------------------------
# SpixSMSTool
# ---------------------------------------------------------------------------


class _SMSInput(BaseModel):
    to: str = Field(
        description="The E.164 phone number to text, e.g. '+19175550123'."
    )
    sender: str = Field(
        description=(
            "The E.164 Spix number to send from, e.g. '+14155550101'. "
            "Must be rented on your account."
        )
    )
    body: str = Field(
        description=(
            "The SMS message body. Keep under 160 characters for a single segment. "
            "Longer messages are split and cost more credits."
        )
    )
    playbook_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional SMS playbook ID. Recommended for agents. "
            "If omitted, resolved automatically from the sender number."
        ),
    )


class SpixSMSTool(BaseTool):
    """Send an SMS via Spix.

    Args:
        api_key: Spix API key. Falls back to the ``SPIX_API_KEY`` env var.

    Example:
        .. code-block:: python

            tool = SpixSMSTool()

            agent = Agent(
                role="Customer Success",
                goal="Send confirmation messages to customers",
                tools=[tool],
            )
    """

    name: str = "Spix SMS"
    description: str = (
        "Send an SMS message via Spix. "
        "Use when you need to text a person. "
        "Required inputs: to (E.164 number), sender (E.164 Spix number), body. "
        "Optional: playbook_id (recommended for agents)."
    )
    args_schema: Type[BaseModel] = _SMSInput
    api_key: Optional[str] = None

    def _run(
        self,
        to: str,
        sender: str,
        body: str,
        playbook_id: Optional[str] = None,
    ) -> str:
        key = _get_api_key(self.api_key)
        payload: dict = {"to": to, "sender": sender, "body": body}
        if playbook_id:
            payload["playbook_id"] = playbook_id
        result = _spix_post("/sms", payload, key)
        message_id = result.get("message_id", "unknown")
        segments = result.get("segments", "?")
        credits_used = result.get("credits_used", "?")
        return (
            f"SMS sent successfully. "
            f"Message ID: {message_id}. "
            f"Segments: {segments}. "
            f"Credits used: {credits_used}."
        )


# ---------------------------------------------------------------------------
# SpixEmailTool
# ---------------------------------------------------------------------------


class _EmailInput(BaseModel):
    sender: str = Field(
        description=(
            "Spix inbox address to send from, e.g. 'support@spix.sh'. "
            "Must be a registered inbox on your Spix account."
        )
    )
    to: str = Field(description="Recipient email address.")
    subject: str = Field(description="Email subject line.")
    body: str = Field(description="Plain-text email body.")


class SpixEmailTool(BaseTool):
    """Send an email via Spix.

    Args:
        api_key: Spix API key. Falls back to the ``SPIX_API_KEY`` env var.

    Example:
        .. code-block:: python

            tool = SpixEmailTool()

            agent = Agent(
                role="Onboarding Specialist",
                goal="Welcome new users via email",
                tools=[tool],
            )
    """

    name: str = "Spix Email"
    description: str = (
        "Send an email via Spix. "
        "Use when you need to email a person. "
        "Required inputs: sender (Spix inbox address), to (recipient email), "
        "subject, body (plain text)."
    )
    args_schema: Type[BaseModel] = _EmailInput
    api_key: Optional[str] = None

    def _run(self, sender: str, to: str, subject: str, body: str) -> str:
        key = _get_api_key(self.api_key)
        result = _spix_post(
            "/email/send",
            {"sender": sender, "to": to, "subject": subject, "body": body},
            key,
        )
        message_id = result.get("message_id", "unknown")
        credits_used = result.get("credits_used", "?")
        return (
            f"Email sent successfully. "
            f"Message ID: {message_id}. "
            f"Credits used: {credits_used}."
        )

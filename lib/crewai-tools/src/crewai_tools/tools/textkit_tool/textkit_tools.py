"""Send SMS and phone verification codes through the TextKit API."""

import asyncio
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


TEXTKIT_API_URL = "https://api.jointextkit.com/v1"


class TextKitSendMessageInput(BaseModel):
    """Input schema for TextKitSendMessageTool."""

    to: str = Field(..., description="Recipient's phone number in E.164 format")
    body: str = Field(..., max_length=1600, description="The message text")
    from_number: str | None = Field(
        default=None,
        description=(
            "Sending number in E.164 format, or a sender pool id. Optional when "
            "the account has exactly one number."
        ),
    )
    idempotency_key: str | None = Field(
        default=None,
        description=(
            "Reuse the same key when retrying so the text is not sent twice. "
            "Valid for 24 hours."
        ),
    )


class TextKitListMessagesInput(BaseModel):
    """Input schema for TextKitListMessagesTool."""

    limit: int = Field(default=25, ge=1, le=100, description="Messages per page")
    cursor: str | None = Field(
        default=None, description="next_cursor from the previous page"
    )
    number: str | None = Field(
        default=None, description="Only messages to or from this number (E.164)"
    )
    status: (
        Literal["queued", "sent", "delivered", "failed", "blocked", "received"] | None
    ) = Field(default=None, description="Only messages with this status")
    direction: Literal["outbound", "inbound"] | None = Field(
        default=None, description="outbound (sent) or inbound (received)"
    )


class TextKitSendVerificationInput(BaseModel):
    """Input schema for TextKitSendVerificationTool."""

    phone: str = Field(..., description="Number to text the code to, in E.164 format")
    app_name: str | None = Field(
        default=None,
        max_length=24,
        description="Product name shown in the message",
    )


class TextKitCheckVerificationInput(BaseModel):
    """Input schema for TextKitCheckVerificationTool."""

    phone: str = Field(..., description="Number the code was texted to, in E.164")
    code: str = Field(..., description="The code the person received")


class TextKitBaseTool(BaseTool):
    """Shared auth, request and error handling for the TextKit tools."""

    timeout: int = Field(default=30, description="Request timeout in seconds")
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="TEXTKIT_API_KEY",
                description="API key for TextKit (tk_sk_...)",
                required=True,
            ),
        ]
    )

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        api_key = os.environ.get("TEXTKIT_API_KEY")
        if not api_key:
            raise ValueError("TEXTKIT_API_KEY environment variable is required")

        headers = {"Authorization": f"Bearer {api_key}"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        try:
            response = requests.request(
                method,
                f"{TEXTKIT_API_URL}{path}",
                headers=headers,
                json=payload,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach the TextKit API: {e}") from e

        if not response.ok:
            raise RuntimeError(self._error_message(response))
        return response.text

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            error = response.json().get("error", {})
            detail = f"{error['code']}: {error['message']}"
        except (AttributeError, KeyError, TypeError, ValueError):
            detail = response.text
        return f"TextKit API error ({response.status_code}): {detail}"

    @staticmethod
    def _without_none(values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}


class TextKitSendMessageTool(TextKitBaseTool):
    """Send an SMS through TextKit."""

    name: str = "textkit_send_message"
    description: str = (
        "Send an SMS to a phone number through TextKit. Texts a real person and "
        "bills the account when used with a live key, so only send when the task "
        "calls for it. Returns the message as JSON, including its id and status."
    )
    args_schema: type[BaseModel] = TextKitSendMessageInput

    def _run(
        self,
        to: str,
        body: str,
        from_number: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        return self._request(
            "POST",
            "/messages",
            payload=self._without_none({"to": to, "body": body, "from": from_number}),
            idempotency_key=idempotency_key,
        )

    async def _arun(
        self,
        to: str,
        body: str,
        from_number: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run, to, body, from_number, idempotency_key
        )


class TextKitListMessagesTool(TextKitBaseTool):
    """List messages on a TextKit account."""

    name: str = "textkit_list_messages"
    description: str = (
        "List SMS messages on a TextKit account, newest first, including replies "
        "received. Returns JSON with data, has_more and next_cursor."
    )
    args_schema: type[BaseModel] = TextKitListMessagesInput

    def _run(
        self,
        limit: int = 25,
        cursor: str | None = None,
        number: str | None = None,
        status: str | None = None,
        direction: str | None = None,
    ) -> str:
        return self._request(
            "GET",
            "/messages",
            params=self._without_none(
                {
                    "limit": limit,
                    "cursor": cursor,
                    "number": number,
                    "status": status,
                    "direction": direction,
                }
            ),
        )

    async def _arun(
        self,
        limit: int = 25,
        cursor: str | None = None,
        number: str | None = None,
        status: str | None = None,
        direction: str | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run, limit, cursor, number, status, direction
        )


class TextKitSendVerificationTool(TextKitBaseTool):
    """Text a one-time verification code through TextKit."""

    name: str = "textkit_send_verification"
    description: str = (
        "Text a one-time verification code to a phone number through TextKit. "
        "No sending number is needed. Follow up with textkit_check_verification."
    )
    args_schema: type[BaseModel] = TextKitSendVerificationInput

    def _run(self, phone: str, app_name: str | None = None) -> str:
        return self._request(
            "POST",
            "/verify",
            payload=self._without_none({"phone": phone, "app_name": app_name}),
        )

    async def _arun(self, phone: str, app_name: str | None = None) -> str:
        return await asyncio.to_thread(self._run, phone, app_name)


class TextKitCheckVerificationTool(TextKitBaseTool):
    """Check a TextKit verification code."""

    name: str = "textkit_check_verification"
    description: str = (
        "Check a code sent by textkit_send_verification. Returns JSON where "
        "verified is true when the code is correct."
    )
    args_schema: type[BaseModel] = TextKitCheckVerificationInput

    def _run(self, phone: str, code: str) -> str:
        return self._request(
            "POST", "/verify/check", payload={"phone": phone, "code": code}
        )

    async def _arun(self, phone: str, code: str) -> str:
        return await asyncio.to_thread(self._run, phone, code)

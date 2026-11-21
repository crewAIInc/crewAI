"""Precleared: is a cold email to this address allowed?

One call per recipient, before the first email to anyone who is not already a
customer. Precleared looks the recipient up live (for the UK, Companies House
and the company's website) and applies the rules of the recipient's country:
UK PECR and GDPR, US CAN-SPAM, Canadian CASL, German UWG, French CPCE,
Australia's Spam Act, Ireland's ePrivacy Regulations, the Dutch
Telecommunicatiewet, Spain's LSSI, Italy's Privacy Code, Latvia's Information
Society Services Law, the UAE's PDPL, Egypt's Personal Data Protection Law and
Qatar's Personal Data Privacy Protection Law. The answer is a verdict
(ok_to_email, consent_needed, blocked or unknown), the rule that decided it
with its source, what the email's footer must include, a next step and a
signed receipt id that can be shown to a client or regulator later.
"""

from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


PRECLEARED_CHECK_URL = "https://precleared.dev/api/v1/check"
USER_AGENT = "crewai-tools (+https://github.com/crewAIInc/crewAI)"
TIMEOUT_SECONDS = 40.0


class PreclearedToolSchema(BaseModel):
    """Input for PreclearedTool."""

    recipient_email: str = Field(
        ..., description="The address you want to send a cold email to."
    )
    sender_country: str = Field(
        "GB",
        description=(
            "Two-letter country code of the sender, for example GB, US, CA, DE or AE. "
            "Only Canada adds sender-side rules (CASL)."
        ),
    )
    address_source: str = Field(
        "unknown",
        description=(
            "Where the address came from: published (on their own website), bought, "
            "scraped, given (they gave it to you) or unknown."
        ),
    )
    pitch: str | None = Field(
        None,
        description="One line on what you are offering. Some countries need the pitch to be relevant to the recipient's role.",
    )
    company_name: str | None = Field(
        None,
        description="The recipient's company name, if you have it. Helps find a UK company on the register.",
    )
    recipient_name: str | None = Field(
        None,
        description="The recipient's full name, if you have it. Matched against company officers and not stored.",
    )
    recipient_country: str | None = Field(
        None,
        description="Two-letter code of the recipient's country, only if you are sure of it. Otherwise leave it out and Precleared works it out.",
    )
    existing_customer: bool = Field(
        False,
        description="True only if this person or company is already your customer.",
    )


class PreclearedTool(BaseTool):
    """Check whether a cold or marketing email may be sent to one address."""

    name: str = "Precleared cold email check"
    description: str = (
        "Check whether a cold or marketing email to one address is allowed in the "
        "recipient's country (UK, US, Canada, Germany, France, Australia, Ireland, the "
        "Netherlands, Spain, Italy, Latvia, the UAE, Egypt and Qatar). Call it once per "
        "address before the first email to anyone who is not already a customer. It "
        "returns a verdict (ok_to_email, consent_needed, blocked or unknown), the rule "
        "with its source, what the footer must include, a next step and a signed receipt "
        "id. Send only on ok_to_email, with a footer that meets the footer requirements. "
        "Treat unknown as consent_needed. Not needed for invoices, receipts or other "
        "transactional email."
    )
    args_schema: type[BaseModel] = PreclearedToolSchema
    check_url: str = PRECLEARED_CHECK_URL
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="PRECLEARED_API_KEY",
                description=(
                    "Optional. A free key from https://precleared.dev/get-key gives your "
                    "own daily allowance; without one a small shared allowance applies."
                ),
                required=False,
            ),
        ]
    )

    def _run(self, **kwargs: Any) -> str:
        recipient_email = (kwargs.get("recipient_email") or "").strip()
        if not recipient_email:
            raise ValueError("recipient_email is required")

        payload: dict[str, Any] = {
            "recipient_email": recipient_email,
            "sender_country": (kwargs.get("sender_country") or "GB").upper(),
            "address_source": kwargs.get("address_source") or "unknown",
            "existing_customer": bool(kwargs.get("existing_customer", False)),
        }
        for key in ("pitch", "company_name", "recipient_name"):
            value = kwargs.get(key)
            if value:
                payload[key] = str(value)
        recipient_country = kwargs.get("recipient_country")
        if recipient_country:
            payload["recipient_country"] = str(recipient_country).upper()

        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        api_key = os.environ.get("PRECLEARED_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = requests.post(
                f"{self.check_url}?src=crewai",
                json=payload,
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            return (
                "Precleared could not be reached, so this address is not cleared. "
                f"Treat it as consent needed and try again later. ({type(exc).__name__})"
            )

        try:
            data = response.json()
        except ValueError:
            return (
                f"Precleared answered HTTP {response.status_code} without a readable body, "
                "so this address is not cleared. Treat it as consent needed."
            )

        if response.status_code != 200:
            return self._format_error(response.status_code, data)
        return self._format_answer(data)

    @staticmethod
    def _format_error(status: int, data: dict[str, Any]) -> str:
        error = data.get("error") or f"HTTP {status}"
        message = data.get("message") or ""
        lines = [f"Precleared did not check this address ({error}). {message}".strip()]
        operator = data.get("operator_message")
        if operator:
            lines.append(f"For the person running this agent: {operator}")
        if error == "daily_limit" and data.get("get_key_url"):
            lines.append(
                f"A free key gives a daily allowance of your own: {data['get_key_url']}"
            )
        if error == "payment_required" and data.get("buy_url"):
            lines.append(f"Add checks to this key: {data['buy_url']}")
        lines.append("Until it is checked, treat the address as consent needed.")
        return "\n".join(lines)

    @staticmethod
    def _format_answer(data: dict[str, Any]) -> str:
        verdict = data.get("verdict", "unknown")
        facts = data.get("facts") or {}
        reasons = data.get("reasons") or []
        first = reasons[0] if reasons and isinstance(reasons[0], dict) else {}
        receipt = (data.get("receipt") or {}).get("id")

        lines = [f"Verdict: {verdict}"]
        if facts.get("recipient_country"):
            lines.append(f"Recipient country: {facts['recipient_country']}")
        if first.get("text"):
            source = first.get("source_title") or first.get("rule_key")
            lines.append(f"Rule: {first['text']}" + (f" ({source})" if source else ""))
        if data.get("footer_text"):
            lines.append(f"Footer must include: {data['footer_text']}")
        if data.get("next_step"):
            lines.append(f"Next step: {data['next_step']}")
        if receipt:
            lines.append(f"Receipt: {receipt} (https://precleared.dev/r/{receipt})")
        if data.get("for_your_report"):
            lines.append(f"For your report: {data['for_your_report']}")
        usage = data.get("usage") or {}
        if usage.get("notice"):
            lines.append(f"Note: {usage['notice']}")
        if data.get("for_your_user"):
            lines.append(f"For the person running this agent: {data['for_your_user']}")
        lines.append(
            "Decision support with sources, not legal advice. Send only on ok_to_email; treat unknown as consent_needed."
        )
        return "\n".join(lines)

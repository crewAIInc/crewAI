from typing import Any, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


class DeepKeepCheckInputSchema(BaseModel):
    """Input schema for DeepKeepCheckInputTool."""

    firewall_id: str = Field(
        ...,
        description="The ID of the DeepKeep firewall to check the input against.",
    )
    conversation_id: str = Field(
        ...,
        description="The ID of the conversation the message belongs to.",
    )
    content: str = Field(
        ...,
        description="The user prompt or text content to be checked by the firewall.",
    )
    logs: bool = Field(
        False,
        description=(
            "Set to True to return the full list of detected violations. "
            "Useful for audit or debugging; defaults to False."
        ),
    )


class DeepKeepCheckInputTool(BaseTool):
    """Check a user prompt against a DeepKeep AI firewall's guardrails.

    Calls POST /v2/firewalls/{firewallId}/conversation/{conversationId}/check_user_input
    and returns the violation details as a JSON string. Use this before passing
    any user input to an LLM to enforce safety policies.

    Requires:
        subdomain: Your DeepKeep tenant subdomain (e.g. "acme").
        api_key:   Your DeepKeep API key.
    """

    name: str = "DeepKeep Check Input"
    description: str = (
        "Check a user prompt or AI-generated text against a DeepKeep AI firewall. "
        "Returns a JSON object with violation details. Use this before passing "
        "any user input to an LLM to enforce guardrails and safety policies."
    )
    args_schema: Type[BaseModel] = DeepKeepCheckInputSchema

    subdomain: str = Field(
        ..., description="Your DeepKeep tenant subdomain (e.g. 'acme')."
    )
    api_key: str = Field(..., description="Your DeepKeep API key.")

    @model_validator(mode="before")
    @classmethod
    def validate_credentials(cls, values: dict[str, Any]) -> dict[str, Any]:
        subdomain = values.get("subdomain", "")
        api_key = values.get("api_key", "")
        if not subdomain:
            raise ValueError("DeepKeepCheckInputTool requires a non-empty 'subdomain'.")
        if not api_key:
            raise ValueError("DeepKeepCheckInputTool requires a non-empty 'api_key'.")
        return values

    def _run(
        self,
        firewall_id: str,
        conversation_id: str,
        content: str,
        logs: bool = False,
    ) -> str:
        import json

        base_url = f"https://api.{self.subdomain}.deepkeep.ai/api"
        url = (
            f"{base_url}/v2/firewalls/{firewall_id}"
            f"/conversation/{conversation_id}/check_user_input"
        )
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        response = requests.post(
            url,
            headers=headers,
            json={"content": content, "logs": logs},
            timeout=30,
        )
        response.raise_for_status()
        return json.dumps({"results": response.json()}, indent=2)

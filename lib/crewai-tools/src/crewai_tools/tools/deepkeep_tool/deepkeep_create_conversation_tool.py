from typing import Any, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


class DeepKeepCreateConversationSchema(BaseModel):
    """Input schema for DeepKeepCreateConversationTool."""

    firewall_id: str = Field(
        ...,
        description="The ID of the DeepKeep firewall in which to create the conversation.",
    )


class DeepKeepCreateConversationTool(BaseTool):
    """Open a new tracked conversation inside a DeepKeep AI firewall.

    Calls POST /v2/firewalls/{firewallId}/conversation and returns the
    conversation object as a JSON string. The conversation ID in the
    response must be passed to DeepKeepCheckInputTool for every subsequent
    message in that conversation.

    Requires:
        subdomain: Your DeepKeep tenant subdomain (e.g. "acme").
        api_key:   Your DeepKeep API key.
    """

    name: str = "DeepKeep Create Conversation"
    description: str = (
        "Create a new conversation inside a DeepKeep AI firewall. "
        "Returns a JSON object containing the conversation ID, which must be "
        "passed to 'DeepKeep Check Input' for every message in that conversation."
    )
    args_schema: Type[BaseModel] = DeepKeepCreateConversationSchema

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
            raise ValueError(
                "DeepKeepCreateConversationTool requires a non-empty 'subdomain'."
            )
        if not api_key:
            raise ValueError(
                "DeepKeepCreateConversationTool requires a non-empty 'api_key'."
            )
        return values

    def _run(self, firewall_id: str) -> str:
        import json

        base_url = f"https://api.{self.subdomain}.deepkeep.ai/api"
        url = f"{base_url}/v2/firewalls/{firewall_id}/conversation"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        # The DeepKeep API requires a JSON body even when empty; sending an
        # empty dict causes some HTTP clients to drop the body, so we send the
        # literal string "{}" with an explicit Content-Type header.
        response = requests.post(url, headers=headers, data="{}", timeout=30)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)

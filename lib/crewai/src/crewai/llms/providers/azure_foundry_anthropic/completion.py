from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from pydantic import model_validator

from crewai.llms.providers.anthropic.completion import AnthropicCompletion


_FOUNDRY_HOST_SUFFIX = ".services.ai.azure.com"


class AzureFoundryAnthropicCompletion(AnthropicCompletion):
    """Claude served by Azure Foundry through Anthropic's Messages API.

    Selected by ``LLM`` when an Azure model name contains ``claude`` or
    ``anthropic``; its provider stays ``azure``.
    """

    provider: str = "azure"
    endpoint: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _read_azure_endpoint(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["endpoint"] = (
                data.get("endpoint")
                or os.getenv("AZURE_ENDPOINT")
                or os.getenv("AZURE_API_BASE")
            )
        return data

    def _get_client_params(self, include_http_client: bool = True) -> dict[str, Any]:
        # Resolved here, not at construction, so the LLM can be built before
        # the endpoint and key are configured; a ValueError defers to the first
        # call. Setting api_key first keeps the parent from falling back to
        # ANTHROPIC_API_KEY, which belongs to api.anthropic.com, not Azure.
        if self.base_url is None:
            self.base_url = self._base_url_from_endpoint()
        if self.api_key is None:
            self.api_key = os.getenv("AZURE_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "AZURE_API_KEY is required for Claude on Foundry. Set it "
                    "or pass api_key=."
                )
        return super()._get_client_params(include_http_client)

    def _base_url_from_endpoint(self) -> str:
        if not self.endpoint:
            raise ValueError(
                "Azure endpoint is required for Claude on Foundry. Set "
                "AZURE_ENDPOINT or pass endpoint= (or base_url=)."
            )
        parsed = urlparse(self.endpoint)
        if not (parsed.hostname or "").endswith(_FOUNDRY_HOST_SUFFIX):
            raise ValueError(
                "Claude on Foundry is served from "
                f"https://<resource>{_FOUNDRY_HOST_SUFFIX}/anthropic, but the "
                f"configured endpoint host is '{parsed.hostname}'. Pass base_url "
                f"explicitly to use a different host."
            )
        return f"https://{parsed.netloc}/anthropic"

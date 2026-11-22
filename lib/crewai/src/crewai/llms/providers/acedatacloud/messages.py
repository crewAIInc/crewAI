"""Ace Messages uses the native adapter without vendor-specific capacity guesses."""

from typing import Any

from anthropic import omit
from pydantic import PrivateAttr

from crewai.llms.base_llm import BaseLLM
from crewai.llms.providers.anthropic.completion import AnthropicCompletion


class AceMessagesCompletion(AnthropicCompletion):
    _ace_vision: bool = PrivateAttr(default=False)

    def _get_client_params(self, include_http_client: bool = True) -> dict[str, Any]:
        params = super()._get_client_params(include_http_client=include_http_client)
        params["default_headers"] = {
            **params.get("default_headers", {}),
            "X-Api-Key": omit,
        }
        return params

    def _supports_native_structured_output(self) -> bool:
        # The compatible host has not declared Anthropic beta structured outputs.
        return False

    def get_context_window_size(self) -> int:
        return BaseLLM.get_context_window_size(self)

    def supports_multimodal(self) -> bool:
        return self._ace_vision

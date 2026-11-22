"""Choose an existing native SDK from the model's published protocol."""

from __future__ import annotations

import os
from typing import Any

from crewai_core.acedatacloud import API_BASE, get_ace_models
from pydantic import PrivateAttr

from crewai.llms.base_llm import BaseLLM, MissingAPIKeyError, _without_placeholder
from crewai.llms.providers.openai_compatible.completion import (
    OpenAICompatibleCompletion,
)


class AceDataCloudCompletion(OpenAICompatibleCompletion):
    """Use Ace metadata, retaining CrewAI's explicit unknown-context budget."""

    _ace_vision: bool = PrivateAttr(default=False)

    def get_context_window_size(self) -> int:
        return BaseLLM.get_context_window_size(self)

    def supports_multimodal(self) -> bool:
        return self._ace_vision


def create_acedatacloud_completion(model: str, **kwargs: Any) -> BaseLLM:
    api_key = _without_placeholder(kwargs.get("api_key")) or _without_placeholder(
        os.getenv("ACEDATACLOUD_API_KEY")
    )
    if not api_key:
        raise MissingAPIKeyError("ACEDATACLOUD_API_KEY", model)
    base = str(
        kwargs.get("base_url")
        or kwargs.get("api_base")
        or os.getenv("ACEDATACLOUD_BASE_URL")
        or API_BASE
    )
    metadata = None
    if base.rstrip("/") == API_BASE:
        metadata = next((item for item in get_ace_models() if item.id == model), None)
    protocol = metadata.protocols[0] if metadata else "chat_completions"
    if kwargs.get("api") == "responses" and metadata:
        if "responses" not in metadata.protocols:
            raise ValueError(f"Responses is not published for Ace model {model}")
        protocol = "responses"
    kwargs.update(api_key=api_key, base_url=base)
    if protocol == "messages":
        # Import only when selected: ordinary chat needs no optional Anthropic SDK.
        from crewai.llms.providers.acedatacloud.messages import AceMessagesCompletion

        kwargs.pop("api", None)
        kwargs.pop("api_base", None)
        kwargs["base_url"] = "https://api.acedata.cloud"
        # This is a request budget, not a claimed model capacity.
        if kwargs.get("max_tokens") is None:
            kwargs["max_tokens"] = 4096
        kwargs["client_params"] = {
            **(kwargs.get("client_params") or {}),
            "api_key": "",
            "auth_token": api_key,
        }
        messages = AceMessagesCompletion(model=model, provider="acedatacloud", **kwargs)
        messages._ace_vision = metadata.vision if metadata else False
        return messages
    if protocol == "responses":
        kwargs["api"] = "responses"
    result = AceDataCloudCompletion(model=model, provider="acedatacloud", **kwargs)
    result._ace_vision = metadata.vision if metadata else False
    return result

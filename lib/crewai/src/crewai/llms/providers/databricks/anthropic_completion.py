"""Claude model services on Databricks Unity Gateway (native Anthropic API).

Calls ``{host}/ai-gateway/anthropic/v1/messages`` with the Anthropic SDK, so the
Claude features crewAI's Anthropic provider supports (tool use, thinking blocks,
prompt caching, streaming) work unchanged. Built by
``LLM(model="databricks/system.ai.<claude...>")`` or ``api="anthropic"``; see
:mod:`crewai.llms.providers.databricks.completion`.
"""

from __future__ import annotations

from typing import Any, Literal

from anthropic import Anthropic, AsyncAnthropic
from pydantic import model_validator

from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.llms.providers.anthropic.completion import (
    AnthropicCompletion,
    _default_max_tokens_for_model,
)
from crewai.llms.providers.databricks.common import (
    DatabricksLLMMixin,
    context_window_aliases,
)
from crewai.llms.providers.databricks.gateway import (
    SYSTEM_AI_PREFIX,
    build_async_httpx_client,
    build_httpx_client,
)
from crewai.llms.providers.utils.common import claude_supports_vision


class DatabricksAnthropicCompletion(DatabricksLLMMixin, AnthropicCompletion):
    """Claude model services on Databricks Unity Gateway (native Messages API)."""

    provider: str = "databricks"
    api: Literal["anthropic"] = "anthropic"

    @model_validator(mode="before")
    @classmethod
    def _normalize_databricks_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["api"] = "anthropic"
            # The base picks the default output limit from a `claude-...` name.
            model = str(data.get("model") or "")
            if data.get("max_tokens") is None and model.startswith(SYSTEM_AI_PREFIX):
                data["max_tokens"] = _default_max_tokens_for_model(
                    context_window_aliases(model)[0]
                )
        return data

    # Replaces AnthropicCompletion._init_clients (its eager client build):
    # clients are built on the first request, because resolving OAuth/profile
    # credentials can take network round trips that must not happen in LLM().
    @model_validator(mode="after")
    def _init_clients(self) -> DatabricksAnthropicCompletion:
        return self

    def _get_client_params(self, include_http_client: bool = True) -> dict[str, Any]:
        # Never reads ANTHROPIC_API_KEY: Databricks Unity Gateway takes the
        # Databricks token, which the auth hook sets (stripping x-api-key).
        client_params: dict[str, Any] = {
            "api_key": "databricks",
            "base_url": self._gateway_url("anthropic"),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if include_http_client:
            transport = (
                HTTPTransport(interceptor=self.interceptor)
                if self.interceptor
                else None
            )
            client_params["http_client"] = build_httpx_client(
                self._databricks_auth(), transport=transport
            )
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    def _build_async_client(self) -> Any:
        params = self._get_client_params(include_http_client=False)
        transport = (
            AsyncHTTPTransport(interceptor=self.interceptor)
            if self.interceptor
            else None
        )
        params.setdefault(
            "http_client",
            build_async_httpx_client(self._databricks_auth(), transport=transport),
        )
        return AsyncAnthropic(**params)

    def _build_sync_client(self) -> Any:
        return Anthropic(**self._get_client_params())

    def get_context_window_size(self) -> int:
        from crewai.llms.context_window import (
            ANTHROPIC_CONTEXT_WINDOWS,
            resolve_context_window_size,
        )

        return resolve_context_window_size(
            self.model,
            ANTHROPIC_CONTEXT_WINDOWS,
            default=200_000,
            extra_names=self._context_window_names(),
            configured=self.context_window_size,
        )

    def supports_multimodal(self) -> bool:
        # The base checks for a `claude-<n>` name; a custom service can have any
        # name, so let Databricks Unity Gateway report what its model accepts.
        name = self._system_model_name()
        return True if name is None else claude_supports_vision(name)

    def _multimodal_formatter_name(self) -> str:
        return "anthropic"

    def get_file_uploader(self) -> Any:
        """Databricks Unity Gateway serves no Anthropic Files API; files are sent inline."""
        return None

    def to_config_dict(self) -> dict[str, Any]:
        config = self._databricks_config_dict(super().to_config_dict())
        name = self._system_model_name()
        if name:
            # The base compares with the default for the full service name;
            # the default this LLM got came from the model name.
            if self.max_tokens == _default_max_tokens_for_model(name):
                config.pop("max_tokens", None)
            else:
                config["max_tokens"] = self.max_tokens
        config["api"] = "anthropic"
        return config

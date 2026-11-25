"""Gemini model services on Databricks Unity Gateway (native Gemini API).

Calls ``{host}/ai-gateway/gemini/v1beta/models/<service>:generateContent`` with
the google-genai SDK. The native API round-trips Gemini thought signatures, so
multi-turn tool calling works; use this API (not the unified Responses API) for
Gemini tool-using agents.
Built by ``LLM(model="databricks/system.ai.<gemini...>")`` or ``api="gemini"``;
see :mod:`crewai.llms.providers.databricks.completion`.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import Field, model_validator

from crewai.llms.hooks.base import BaseInterceptor
from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.llms.providers.databricks.common import (
    DatabricksLLMMixin,
    context_window_aliases,
)
from crewai.llms.providers.databricks.gateway import (
    build_async_httpx_client,
    build_httpx_client,
)
from crewai.llms.providers.gemini.completion import GeminiCompletion


class DatabricksGeminiCompletion(DatabricksLLMMixin, GeminiCompletion):
    """Gemini model services on Databricks Unity Gateway (native Gemini API)."""

    provider: str = "databricks"
    api: Literal["gemini"] = "gemini"
    # Holds `interceptor` past the base validator, which rejects interceptors
    # because its client owns the HTTP layer; here the httpx clients are ours.
    gateway_interceptor: BaseInterceptor[Any, Any] | None = Field(
        default=None, exclude=True, repr=False
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_databricks_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data["api"] = "gemini"
            data["use_vertexai"] = False
            # One output limit across Databricks model services: Claude and the
            # unified APIs take `max_tokens`, so honour it here too.
            if data.get("max_output_tokens") is None and data.get("max_tokens"):
                data["max_output_tokens"] = int(data["max_tokens"])
            if data.get("interceptor") is not None:
                data["gateway_interceptor"] = data.pop("interceptor")
            # The base turns thinking on for `gemini-2.5`+ names; Databricks
            # names write versions with a dash (`gemini-2-5-flash`).
            dotted = context_window_aliases(str(data.get("model") or ""))[-1]
            version = re.search(r"gemini-(\d+(?:\.\d+)?)", dotted.lower())
            if (
                data.get("thinking_config") is None
                and version
                and float(version.group(1)) >= 2.5
            ):
                data["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        return data

    @model_validator(mode="after")
    def _gateway_capabilities(self) -> DatabricksGeminiCompletion:
        # The base infers tool support from a `gemini-X.Y` model name; a model
        # service can have any name, and Databricks Unity Gateway Gemini models
        # all take tools.
        self.supports_tools = True
        self.is_gemini_2_0 = True
        if self.gateway_interceptor is not None:
            self.interceptor = self.gateway_interceptor
            self.gateway_interceptor = None
        return self

    # Replaces GeminiCompletion._init_client (its eager client build): the
    # client is built on the first request, because resolving OAuth/profile
    # credentials can take network round trips that must not happen in LLM().
    @model_validator(mode="after")
    def _init_client(self) -> DatabricksGeminiCompletion:
        return self

    def _initialize_client(self, use_vertexai: bool = False) -> genai.Client:
        params: dict[str, Any] = dict(self.client_params or {})
        # The SDK insists on a key; the Databricks auth hook replaces it with the
        # bearer token and strips x-goog-api-key.
        params.setdefault("api_key", "databricks")
        # Never Vertex AI, whatever GOOGLE_GENAI_USE_VERTEXAI says.
        params.setdefault("vertexai", False)
        if "http_options" not in params:
            auth = self._databricks_auth()
            interceptor = self.interceptor
            params["http_options"] = types.HttpOptions(
                base_url=self._gateway_url("gemini"),
                httpx_client=build_httpx_client(
                    auth,
                    transport=HTTPTransport(interceptor=interceptor)
                    if interceptor
                    else None,
                ),
                httpx_async_client=build_async_httpx_client(
                    auth,
                    transport=AsyncHTTPTransport(interceptor=interceptor)
                    if interceptor
                    else None,
                ),
            )
        return genai.Client(**params)

    def _get_sync_client(self) -> Any:
        # The base re-reads GOOGLE_API_KEY here; Databricks never uses it.
        if self._client is None:
            self._client = self._initialize_client()
        return self._client

    def get_context_window_size(self) -> int:
        from crewai.llms.context_window import (
            GEMINI_CONTEXT_WINDOWS,
            resolve_context_window_size,
        )

        return resolve_context_window_size(
            self.model,
            GEMINI_CONTEXT_WINDOWS,
            default=1_048_576,
            extra_names=self._context_window_names(),
            configured=self.context_window_size,
        )

    def _multimodal_formatter_name(self) -> str:
        return "gemini"

    def get_file_uploader(self) -> Any:
        """Databricks Unity Gateway serves no Gemini Files API; files are sent inline."""
        return None

    def to_config_dict(self) -> dict[str, Any]:
        config = self._databricks_config_dict(super().to_config_dict())
        config["api"] = "gemini"
        config.pop("project", None)
        config.pop("location", None)
        return config

"""Databricks LLM provider: Databricks Unity Gateway model services.

``LLM(model="databricks/<model-service>")`` calls a Databricks Unity Gateway
model service: a Databricks-provided ``system.ai.<model>`` or one you configured
in Unity Catalog (``<catalog>.<schema>.<service>``). The API is picked from the
service name, with no network I/O, or from ``api=``:

=========================================  ==================================  =============================
Model service                              Databricks Unity Gateway API        Class
=========================================  ==================================  =============================
``system.ai.*claude*``                     native Anthropic Messages           DatabricksAnthropicCompletion
``system.ai.*gemini*``                     native Gemini ``generateContent``   DatabricksGeminiCompletion
any other ``system.ai.*``, any custom one  unified Responses                   DatabricksCompletion
any, with ``api="completions"``            unified Chat Completions            DatabricksCompletion
=========================================  ==================================  =============================

A custom ``<catalog>.<schema>.<service>`` backed by Claude or Gemini can use its
native API with ``api="anthropic"`` / ``api="gemini"`` (needed for Gemini
multi-turn tool calling).

Authentication uses Databricks unified auth, resolved on the first request:

* personal access token: ``api_key=`` (or ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``)
* OAuth M2M: ``client_id=`` / ``client_secret=`` (or ``DATABRICKS_CLIENT_ID`` /
  ``DATABRICKS_CLIENT_SECRET``)
* OAuth U2M / CLI profile: ``profile=`` (or ``DATABRICKS_CONFIG_PROFILE``), e.g. a
  ``databricks auth login`` profile; ``auth_type="external-browser"`` signs in
  interactively
* an existing ``databricks_config=`` / ``workspace_client=``

Tokens are refreshed per request. Requests carry the ``crewai_oss/<version>``
User-Agent (Databricks requests made by this provider only). Example::

    from crewai import LLM

    claude = LLM(model="databricks/system.ai.claude-sonnet-5")
    support = LLM(model="databricks/main.ai.support_bot", profile="prod")
    gemini_svc = LLM(model="databricks/main.ai.gemini_service", api="gemini")
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import model_validator

from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport
from crewai.llms.providers.databricks.common import DatabricksLLMMixin
from crewai.llms.providers.databricks.gateway import (
    build_async_httpx_client,
    build_httpx_client,
    native_family,
)
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.utils.common import claude_supports_vision


if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM


logger = logging.getLogger(__name__)

DATABRICKS_APIS: Final[tuple[str, ...]] = (
    "auto",
    "anthropic",
    "gemini",
    "responses",
    "completions",
)

# Responses-API features that need server-side response storage, which the
# gateway does not provide (it rejects `previous_response_id`).
_UNSUPPORTED_RESPONSES_OPTIONS: Final[tuple[str, ...]] = (
    "auto_chain",
    "auto_chain_reasoning",
    "previous_response_id",
    "store",
)


def _native_class(family: str) -> type[BaseLLM] | None:
    """The native-API class for ``family``, or None when its SDK isn't installed."""
    try:
        if family == "anthropic":
            from crewai.llms.providers.databricks.anthropic_completion import (
                DatabricksAnthropicCompletion,
            )

            return DatabricksAnthropicCompletion
        if family == "gemini":
            from crewai.llms.providers.databricks.gemini_completion import (
                DatabricksGeminiCompletion,
            )

            return DatabricksGeminiCompletion
    except ImportError as error:
        logger.warning(
            "The %s SDK is not installed (%s), so the Databricks model service is "
            "called through the unified Responses API instead. Install it with: "
            "uv add 'crewai[databricks]'",
            family,
            error,
        )
    return None


def _is_remap() -> bool:
    from crewai.llm_overlay import is_building_mapped_model

    return is_building_mapped_model()


def databricks_llm_class(model: str, api: str | None = None) -> type[BaseLLM]:
    """The Databricks LLM class for ``model`` and ``api`` (pure; no I/O).

    Raises ``ValueError`` when ``api`` names a native API that a Databricks-provided
    ``system.ai.*`` service cannot serve (e.g. ``api="anthropic"`` for a GPT model).
    """
    choice = api or "auto"
    if choice != "auto" and _is_remap():
        # A model the overlay remapped carries the previous model's `api`; it
        # describes that model, so route the new one by its own name.
        choice = "auto"
    family = native_family(model) if choice == "auto" else choice
    if (
        choice in ("anthropic", "gemini")
        and model.startswith("system.ai.")
        and native_family(model) != choice
    ):
        raise ValueError(
            f"{model!r} is not served by the native {choice} API; omit api= or "
            "use api='responses' / api='completions'."
        )
    if family in ("anthropic", "gemini"):
        native = _native_class(family)
        if native is not None:
            return native
    return DatabricksCompletion


class DatabricksCompletion(DatabricksLLMMixin, OpenAICompletion):
    """Databricks model services through the Databricks Unity Gateway unified APIs.

    Uses ``{host}/ai-gateway/mlflow/v1`` with the OpenAI SDK: the Responses API
    by default (``api="responses"``), or Chat Completions (``api="completions"``).
    Build it with ``LLM(model="databricks/<service>")`` to get the native
    Anthropic/Gemini class where that fits.
    """

    provider: str = "databricks"
    api: Literal["responses", "completions"] = "responses"

    @model_validator(mode="before")
    @classmethod
    def _normalize_databricks_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        api = data.get("api") or "responses"
        if api in ("auto", "anthropic", "gemini"):
            # No native API applies, or its SDK isn't installed.
            api = "responses"
        if api not in ("responses", "completions"):
            raise ValueError(
                f"Unsupported Databricks api {api!r}; use one of: "
                + ", ".join(DATABRICKS_APIS)
            )
        data["api"] = api
        for option in _UNSUPPORTED_RESPONSES_OPTIONS:
            if data.get(option):
                raise ValueError(
                    f"{option} is not supported on Databricks Unity Gateway: it does "
                    "not store responses or accept previous_response_id. Pass the "
                    "full conversation instead."
                )
        return data

    # Replaces OpenAICompletion._init_clients (its eager client build): clients
    # are built on the first request, because resolving OAuth/profile
    # credentials can take network round trips that must not happen in LLM().
    @model_validator(mode="after")
    def _init_clients(self) -> DatabricksCompletion:
        return self

    def _get_client_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            # The SDK insists on a key; the Databricks auth hook replaces it.
            "api_key": "databricks",
            "base_url": self._gateway_url("mlflow"),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        client_params = {
            key: value for key, value in params.items() if value is not None
        }
        if self.client_params:
            client_params.update(self.client_params)
        return client_params

    def _build_sync_client(self) -> Any:
        from openai import OpenAI

        params = self._get_client_params()
        transport = (
            HTTPTransport(interceptor=self.interceptor) if self.interceptor else None
        )
        params.setdefault(
            "http_client",
            build_httpx_client(self._databricks_auth(), transport=transport),
        )
        return OpenAI(**params)

    def _build_async_client(self) -> Any:
        from openai import AsyncOpenAI

        params = self._get_client_params()
        transport = (
            AsyncHTTPTransport(interceptor=self.interceptor)
            if self.interceptor
            else None
        )
        params.setdefault(
            "http_client",
            build_async_httpx_client(self._databricks_auth(), transport=transport),
        )
        return AsyncOpenAI(**params)

    def _effective_base_url(self) -> str | None:
        # Keys what the base learns about a model (e.g. that it rejects
        # `reasoning_effort`) by workspace, not by api.openai.com. As in
        # `_get_client_params`, a `base_url` in `client_params` wins.
        override = (self.client_params or {}).get("base_url")
        return str(override) if override else self._gateway_url("mlflow")

    def supports_multimodal(self) -> bool:
        # The base checks the name for `gpt-4o`, `gpt-5`, ...; the unified API
        # serves every model family, and a custom service can have any name,
        # so let Databricks Unity Gateway report what its model accepts.
        name = self._system_model_name()
        if name is None:
            return True
        family = native_family(self.model)
        if family == "anthropic":
            return claude_supports_vision(name)
        if re.match(r"(gpt-|o\d)", name):
            return OpenAICompletion.supports_multimodal(
                self.model_copy(update={"model": name})
            )
        # Gemini, and model families the name checks don't know.
        return True

    def _multimodal_formatter_name(self) -> str:
        return "openai"

    def _on_compatible_server(self) -> bool:
        # Databricks Unity Gateway is an OpenAI-compatible server, whatever its
        # URL resolves to.
        return True

    def get_context_window_size(self) -> int:
        from crewai.llms.context_window import (
            DEFAULT_CONTEXT_WINDOW_SIZE,
            OPENAI_CONTEXT_WINDOWS,
            resolve_context_window_size,
        )

        return resolve_context_window_size(
            self.model,
            OPENAI_CONTEXT_WINDOWS,
            default=DEFAULT_CONTEXT_WINDOW_SIZE,
            extra_names=self._context_window_names(),
            configured=self.context_window_size,
        )

    def get_file_uploader(self) -> Any:
        """Databricks Unity Gateway serves no Files API; files are sent inline."""
        return None

    def to_config_dict(self) -> dict[str, Any]:
        config = self._databricks_config_dict(super().to_config_dict())
        config["api"] = self.api
        return config

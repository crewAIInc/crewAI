"""Behaviour shared by the Databricks LLM classes (one per gateway API)."""

from __future__ import annotations

import asyncio
import re
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from crewai.llms.base_llm import _without_placeholder
from crewai.llms.providers.databricks.auth import DatabricksAuth, require_secure_url
from crewai.llms.providers.databricks.gateway import (
    SYSTEM_AI_PREFIX,
    GatewaySurface,
    gateway_base_url,
)


def context_window_aliases(name: str) -> list[str]:
    """Names a catalog may list a gateway model under.

    ``system.ai.gpt-5-4-mini`` -> ``gpt-5-4-mini``, ``gpt-5.4-mini``.
    """
    bare = name.removeprefix(SYSTEM_AI_PREFIX).removeprefix("databricks-")
    aliases = [bare, re.sub(r"(\d)-(\d)", r"\1.\2", bare)]
    return [
        alias for index, alias in enumerate(aliases) if alias not in aliases[:index]
    ]


class DatabricksLLMMixin(BaseModel):
    """Databricks credentials and helpers mixed into each gateway LLM class.

    ``api_key`` is taken as a Databricks personal access token. Every other
    credential source is resolved by :class:`DatabricksAuth` (SDK unified auth)
    on the first request, so building an LLM never does network I/O.

    The credentials are fixed when the LLM is built and shared by its copies;
    to use other credentials (another host, profile or token), build a new LLM.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    host: str | None = Field(
        default=None, description="Workspace URL; defaults to DATABRICKS_HOST/profile."
    )
    profile: str | None = Field(
        default=None, description="Databricks CLI profile (~/.databrickscfg)."
    )
    auth_type: str | None = Field(
        default=None, description="Force an SDK auth type, e.g. 'external-browser'."
    )
    client_id: str | None = Field(default=None, description="OAuth client ID.")
    client_secret: SecretStr | None = Field(
        default=None, exclude=True, repr=False, description="OAuth client secret."
    )
    databricks_token: SecretStr | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Personal access token (also accepted as api_key).",
    )
    databricks_config: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description="A databricks.sdk.core.Config to authenticate with.",
    )
    workspace_client: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description="A databricks.sdk.WorkspaceClient to authenticate with.",
    )
    databricks_auth: Any = Field(
        default=None,
        exclude=True,
        repr=False,
        description="A prebuilt DatabricksAuth (to share credentials between LLMs).",
    )

    # Runs before the base provider's validators (most-derived first), so a base
    # that falls back to its vendor key (GOOGLE_API_KEY, ...) never sees the PAT
    # in api_key and the PAT is never mistaken for a vendor key.
    @model_validator(mode="before")
    @classmethod
    def _take_databricks_token(cls, data: Any) -> Any:
        if isinstance(data, dict):
            token = _without_placeholder(data.pop("api_key", None))
            if token and not data.get("databricks_token"):
                data["databricks_token"] = token
            # `api_base` is accepted as an alias of `base_url`, as elsewhere in crewAI.
            api_base = data.pop("api_base", None)
            if api_base and not data.get("base_url"):
                data["base_url"] = api_base
            data["provider"] = "databricks"
        return data

    @model_validator(mode="after")
    def _drop_vendor_key(self) -> Any:
        # Whatever a base filled in from its vendor environment is never used.
        self.api_key = None
        # Built now (no I/O: credentials resolve on the first request), so
        # copies of this LLM (agents, crews, kickoff_for_each) share one set of
        # credentials and one SDK token cache.
        if not isinstance(self.databricks_auth, DatabricksAuth):
            self._create_databricks_auth()
        return self

    # Databricks Unity Gateway serves no vendor Files API: files go inline.
    _inline_files_only: ClassVar[bool] = True

    def file_constraints_provider(self) -> str:
        # File-type support follows the API's content-block format, not the
        # "databricks" provider name.
        return self._multimodal_formatter_name()  # type: ignore[attr-defined, no-any-return]

    # -- model name ---------------------------------------------------------------
    def _system_model_name(self) -> str | None:
        """The model a Databricks-provided service names, or ``None``.

        ``system.ai.claude-sonnet-5`` -> ``claude-sonnet-5``. A custom service
        name says nothing about the model behind it, so it gets ``None``.
        """
        model: str = self.model  # type: ignore[attr-defined]
        if not model.startswith(SYSTEM_AI_PREFIX):
            return None
        return context_window_aliases(model)[0]

    # -- auth ---------------------------------------------------------------------
    def _databricks_auth(self) -> DatabricksAuth:
        auth = self.databricks_auth
        if isinstance(auth, DatabricksAuth):
            return auth
        return self._create_databricks_auth()

    def _create_databricks_auth(self) -> DatabricksAuth:
        auth = DatabricksAuth(
            host=self.host,
            token=self.databricks_token.get_secret_value()
            if self.databricks_token
            else None,
            profile=self.profile,
            auth_type=self.auth_type,
            client_id=self.client_id,
            client_secret=(
                self.client_secret.get_secret_value() if self.client_secret else None
            ),
            config=self.databricks_config,
            workspace_client=self.workspace_client,
        )
        # Credentials now live only in the shared, copy-safe DatabricksAuth.
        self.databricks_auth = auth
        self.databricks_config = None
        self.workspace_client = None
        self.client_secret = None
        self.databricks_token = None
        return auth

    def _gateway_url(self, surface: GatewaySurface) -> str:
        explicit = getattr(self, "base_url", None)
        if explicit:
            url = require_secure_url(str(explicit).rstrip("/"))
            if httpx.URL(url).path in ("", "/"):
                # A bare host: add this API's Databricks Unity Gateway path.
                return gateway_base_url(url, surface)
            return url
        return gateway_base_url(self._databricks_auth().host, surface)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        """Resolve credentials off the event loop, then call the provider.

        Resolving OAuth/profile credentials can take blocking network round
        trips (the Databricks SDK fetches host metadata), which must not run on
        the event loop when the async client is first built.
        """
        auth = self._databricks_auth()
        if not auth.is_resolved:
            await asyncio.to_thread(lambda: auth.host)
        return await super().acall(*args, **kwargs)  # type: ignore[misc]

    def _context_window_names(self) -> list[str]:
        return context_window_aliases(self.model)  # type: ignore[attr-defined]

    def _databricks_config_dict(self, config: dict[str, Any]) -> dict[str, Any]:
        """Serialize without secrets, so ``LLM(**config)`` rebuilds this LLM."""
        config["model"] = f"databricks/{self.model}"  # type: ignore[attr-defined]
        config.pop("custom_openai", None)
        for name in ("host", "profile", "auth_type", "client_id"):
            value = getattr(self, name)
            if value:
                config[name] = value
        if "host" not in config and isinstance(self.databricks_auth, DatabricksAuth):
            host = self.databricks_auth.resolved_host
            if host:
                config["host"] = host
        return config

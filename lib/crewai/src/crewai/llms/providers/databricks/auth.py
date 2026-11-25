"""Databricks unified authentication for crewAI.

One resolver for every Databricks call the Databricks provider makes. It delegates to the Databricks SDK's unified auth
(``databricks.sdk.core.Config``), so every method the SDK supports works the
same way everywhere, and tokens are cached and refreshed by the SDK:

=========================  ==================================================
Auth method                How to select it
=========================  ==================================================
Personal access token      ``api_key=``/``token=``, or ``DATABRICKS_TOKEN``
OAuth M2M (service princ.) ``client_id=`` + ``client_secret=``, or
                           ``DATABRICKS_CLIENT_ID`` / ``DATABRICKS_CLIENT_SECRET``
OAuth U2M / CLI profile    ``profile=`` or ``DATABRICKS_CONFIG_PROFILE``
                           (e.g. a ``databricks auth login`` profile), or
                           ``auth_type="external-browser"``
Anything else the SDK has  ``auth_type=`` / ``databricks_config=`` (an SDK
                           ``Config``) / ``workspace_client=``
=========================  ==================================================

Without ``databricks-sdk`` installed only a PAT (+ host) works; every other
method raises a clear install hint.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
import logging
import os
import threading
from typing import Any

import httpx

from crewai.llms.providers.databricks.telemetry import (
    USER_AGENT_PRODUCT,
    user_agent_version,
    with_user_agent,
)


logger = logging.getLogger(__name__)

_INSTALL_HINT = "Install it with: uv add 'crewai[databricks]'"

# Vendor API-key headers the client SDKs add on their own. Databricks Unity Gateway authenticates
# with the Databricks bearer token only, so these must never reach it.
_VENDOR_KEY_HEADERS = ("x-api-key", "x-goog-api-key")


class DatabricksAuthError(ValueError):
    """Databricks credentials could not be resolved or used.

    A ``ValueError`` so LLM providers defer client construction (as they do for a
    missing API key) when credentials aren't available yet at import time.
    """


_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def require_secure_url(url: str) -> str:
    """Return ``url`` if it is ``https://`` (or ``http://`` to a local test server).

    Bearer tokens must not travel in cleartext.
    """
    scheme = url.split("://", 1)[0].lower() if "://" in url else ""
    if scheme != "https" and not (
        scheme == "http" and httpx.URL(url).host in _LOCAL_HOSTS
    ):
        raise DatabricksAuthError(
            f"Databricks URLs must use https:// (got {url!r}); credentials are "
            "never sent over an unencrypted connection."
        )
    return url


def normalize_host(host: str) -> str:
    """Return ``host`` as ``https://<host>`` without a trailing slash or API path."""
    value = host.strip().rstrip("/")
    if not value:
        raise DatabricksAuthError("Databricks host cannot be empty")
    if "://" not in value:
        value = f"https://{value}"
    require_secure_url(value)
    for marker in ("/ai-gateway", "/serving-endpoints", "/api/"):
        index = value.find(marker)
        if index != -1:
            value = value[:index]
    return value.rstrip("/")


def _sanitize_sdk_error(error: BaseException) -> str:
    """The SDK appends a ``Config: host=..., client_id=...`` dump; keep only the cause."""
    message = str(error).strip()
    for marker in (". Config:", " Config:"):
        index = message.find(marker)
        if index != -1:
            message = message[:index]
    return message


class DatabricksAuth:
    """Resolve Databricks credentials once and hand out fresh auth headers.

    Construction never performs I/O; credentials are resolved on first use.
    Instances are shared (``copy``/``deepcopy`` return ``self``) so copying an
    agent or LLM never duplicates token caches or SDK clients.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        token: str | None = None,
        profile: str | None = None,
        auth_type: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        config: Any | None = None,
        workspace_client: Any | None = None,
    ) -> None:
        self._host = normalize_host(host) if host else None
        self._token = token or None
        self._profile = profile or None
        self._auth_type = auth_type or None
        self._client_id = client_id or None
        self._client_secret = client_secret or None
        if config is None and workspace_client is not None:
            config = workspace_client.config
        self._config: Any | None = config
        self._static = False
        self._resolved = config is not None
        self._lock = threading.Lock()

    # -- resolution ---------------------------------------------------------------
    def _resolve(self) -> None:
        if self._resolved:
            return
        with self._lock:
            if self._resolved:
                return
            if self._token and self._auth_type in (None, "pat"):
                # An explicit token wins over whatever the environment or a
                # profile would select, and needs no SDK round trip (the SDK
                # fetches host metadata when it builds a Config).
                self._resolve_static_token()
                return
            if self._env_pat_only():
                self._token = os.environ["DATABRICKS_TOKEN"]
                self._resolve_static_token()
                return
            try:
                from databricks.sdk.core import Config
            except ImportError:
                self._resolve_without_sdk()
                return

            kwargs: dict[str, Any] = {
                "host": self._host,
                "profile": self._profile,
                "auth_type": self._auth_type,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "token": self._token,
                "product": USER_AGENT_PRODUCT,
                "product_version": user_agent_version(),
            }
            try:
                self._config = Config(
                    **{key: value for key, value in kwargs.items() if value is not None}
                )
            except Exception as error:
                raise DatabricksAuthError(
                    "Databricks authentication could not be configured: "
                    f"{_sanitize_sdk_error(error)}"
                ) from error
            self._resolved = True

    def _env_pat_only(self) -> bool:
        """``DATABRICKS_TOKEN`` (+ host) is the only credential configured anywhere."""
        if any((self._profile, self._auth_type, self._client_id, self._client_secret)):
            return False
        if not (
            os.getenv("DATABRICKS_TOKEN")
            and (self._host or os.getenv("DATABRICKS_HOST"))
        ):
            return False
        return not any(
            os.getenv(name)
            for name in (
                "DATABRICKS_CLIENT_ID",
                "DATABRICKS_CLIENT_SECRET",
                "DATABRICKS_CONFIG_PROFILE",
                "DATABRICKS_AUTH_TYPE",
            )
        )

    def _resolve_static_token(self) -> None:
        host = self._host or os.getenv("DATABRICKS_HOST")
        if not host:
            raise DatabricksAuthError(
                "Databricks host is not configured. Set DATABRICKS_HOST or pass host=."
            )
        self._host = normalize_host(host)
        self._static = True
        self._resolved = True

    def _resolve_without_sdk(self) -> None:
        token = self._token or os.getenv("DATABRICKS_TOKEN")
        host = self._host or os.getenv("DATABRICKS_HOST")
        oauth_requested = any(
            (self._profile, self._auth_type, self._client_id, self._client_secret)
        ) or any(
            os.getenv(name)
            for name in (
                "DATABRICKS_CLIENT_ID",
                "DATABRICKS_CONFIG_PROFILE",
                "DATABRICKS_AUTH_TYPE",
            )
        )
        if token and host and not oauth_requested:
            self._token = token
            self._host = normalize_host(host)
            self._static = True
            self._resolved = True
            return
        raise DatabricksAuthError(
            "databricks-sdk is required for Databricks OAuth (M2M/U2M) and "
            f"CLI-profile authentication. {_INSTALL_HINT}. Without it, set "
            "DATABRICKS_HOST and DATABRICKS_TOKEN (a personal access token)."
        )

    # -- public API ---------------------------------------------------------------
    @property
    def host(self) -> str:
        """Workspace URL, e.g. ``https://dbc-123.cloud.databricks.com``."""
        self._resolve()
        host = self._host if self._static else getattr(self._config, "host", None)
        if not host:
            raise DatabricksAuthError(
                "Databricks host is not configured. Set DATABRICKS_HOST, pass "
                "host=, or select a profile that defines it."
            )
        return normalize_host(host)

    def auth_headers(self) -> dict[str, str]:
        """Return fresh auth headers (``Authorization: Bearer ...``) for one request."""
        self._resolve()
        if self._static:
            return {"Authorization": f"Bearer {self._token}"}
        try:
            headers = self._config.authenticate()  # type: ignore[union-attr]
        except Exception as error:
            raise DatabricksAuthError(
                f"Databricks authentication failed ({self.describe()}): "
                f"{_sanitize_sdk_error(error)}"
            ) from error
        return dict(headers)

    @property
    def is_static(self) -> bool:
        """Whether credentials are a resolved static token (no I/O to sign)."""
        return self._resolved and self._static

    @property
    def is_resolved(self) -> bool:
        """Whether credentials have been resolved (no I/O left to find the host)."""
        return self._resolved

    @property
    def resolved_host(self) -> str | None:
        """The workspace URL if credentials are already resolved, else ``None`` (no I/O)."""
        if not self._resolved:
            return self._host
        try:
            return self.host
        except DatabricksAuthError:
            return None

    def describe(self) -> str:
        """Human-readable auth method, safe to log (never includes secrets)."""
        if not self._resolved:
            return "unresolved"
        if self._static:
            return "auth_type=pat"
        auth_type = getattr(self._config, "auth_type", None) or "auto"
        profile = getattr(self._config, "profile", None)
        return f"auth_type={auth_type}" + (f", profile={profile}" if profile else "")

    def httpx_auth(self) -> DatabricksHttpxAuth:
        """An ``httpx.Auth`` that signs every request with a fresh token."""
        return DatabricksHttpxAuth(self)

    def __copy__(self) -> DatabricksAuth:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> DatabricksAuth:
        return self

    def __repr__(self) -> str:
        return f"DatabricksAuth({self.describe()})"


class DatabricksHttpxAuth(httpx.Auth):
    """Sets a fresh Databricks bearer token and crewAI's User-Agent on every request.

    Used by every Databricks Unity Gateway client (openai, anthropic,
    google-genai), so token refresh and the User-Agent behave the same whichever
    API a model service uses.
    """

    def __init__(self, auth: DatabricksAuth) -> None:
        self._auth = auth

    def _apply(self, request: httpx.Request, headers: dict[str, str]) -> None:
        for name in _VENDOR_KEY_HEADERS:
            request.headers.pop(name, None)
        for name, value in headers.items():
            request.headers[name] = value
        request.headers["User-Agent"] = with_user_agent(
            request.headers.get("User-Agent")
        )

    def sync_auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        self._apply(request, self._auth.auth_headers())
        yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if self._auth.is_static:
            headers = self._auth.auth_headers()
        else:
            # The SDK refresh can be blocking I/O; keep it off the event loop.
            headers = await asyncio.to_thread(self._auth.auth_headers)
        self._apply(request, headers)
        yield request

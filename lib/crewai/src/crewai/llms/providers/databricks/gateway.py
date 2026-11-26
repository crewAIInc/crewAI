"""Databricks Unity Gateway helpers: base URLs, API choice, and the HTTP clients.

Model services (``system.ai.<model>`` or a customer-configured
``<catalog>.<schema>.<service>``) are served by three Databricks Unity Gateway APIs:

* ``{host}/ai-gateway/anthropic`` -- native Anthropic Messages (Claude)
* ``{host}/ai-gateway/gemini``    -- native Gemini ``generateContent``
* ``{host}/ai-gateway/mlflow/v1`` -- unified OpenAI-compatible APIs for every model

Every client is built by :func:`build_httpx_client` / :func:`build_async_httpx_client`.
Their transport handles two Databricks Unity Gateway behaviours for all three
client SDKs at once:

* upstream errors arrive wrapped as
  ``{"error_code": ..., "message": "<upstream JSON as a string>"}``; the transport
  hands the SDK the upstream JSON, so its own error handling (retries,
  ``reasoning_effort`` recovery, error types) sees the shape it expects;
* some models reject ``temperature`` (Claude Sonnet 5, GPT-6); the transport
  resends that request once without it and omits it for later requests.

Proxies are honoured as httpx's default client would: ``HTTP_PROXY`` /
``HTTPS_PROXY`` / ``ALL_PROXY`` / ``NO_PROXY``, or the macOS/Windows system proxy
settings when none are set.

An interceptor (crewAI's ``interceptor=``) runs inside this transport, so it sees
the actual wire traffic: Databricks Unity Gateway's error envelope as received,
and the extra request sent when a ``temperature`` is retried. As with other
providers, its transport replaces the proxy handling above.
"""

from __future__ import annotations

from collections.abc import Callable
import ipaddress
import json
import logging
import re
import threading
import time
from typing import Any, Final, Literal
import urllib.request

import httpx

from crewai.llms.providers.databricks.auth import DatabricksAuth, normalize_host


logger = logging.getLogger(__name__)

GatewaySurface = Literal["mlflow", "anthropic", "gemini"]
NativeFamily = Literal["anthropic", "gemini"]

GATEWAY_PATHS: Final[dict[str, str]] = {
    "mlflow": "/ai-gateway/mlflow/v1",
    "anthropic": "/ai-gateway/anthropic",
    "gemini": "/ai-gateway/gemini",
}

SYSTEM_AI_PREFIX: Final[str] = "system.ai."

DOCS_URL: Final[str] = "https://docs.crewai.com/en/concepts/llms"

_DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


def gateway_base_url(host: str, surface: GatewaySurface) -> str:
    """Return the Databricks Unity Gateway base URL for ``surface`` on ``host``."""
    return f"{normalize_host(host)}{GATEWAY_PATHS[surface]}"


def native_family(model: str) -> NativeFamily | None:
    """The native API a Databricks-provided ``system.ai.*`` service uses, if any.

    Databricks-provided services name their model. A custom service name says
    nothing about the model behind it, so it gets ``None`` (the unified API, which
    serves every model) unless the caller picks a native API with ``api=``.
    """
    if not model.startswith(SYSTEM_AI_PREFIX):
        return None
    lowered = model.lower()
    if "claude" in lowered:
        return "anthropic"
    if "gemini" in lowered:
        return "gemini"
    return None


def unwrap_gateway_error_body(body: Any) -> Any:
    """Return the upstream provider error Databricks wraps, or ``body`` unchanged."""
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str) and message.lstrip().startswith("{"):
            try:
                inner = json.loads(message)
            except ValueError:
                return body
            if isinstance(inner, dict):
                return inner
    return body


# -- process state (log-once memory, temperature memory) ---------------------------------

#: Monotonic clock used for the temperature memory (replaceable in tests).
clock: Callable[[], float] = time.monotonic

_state_lock = threading.Lock()
_reported_problems: set[tuple[str, str]] = set()
# (host, model) -> expiry for models that rejected `temperature`, so the
# rejected round trip is paid about once an hour rather than on every call.
# Expiring lets a service re-pointed to another model recover.
_no_temperature: dict[tuple[str, str], float] = {}


def reset_state() -> None:
    """Forget remembered temperature rejections and already-logged problems."""
    with _state_lock:
        _reported_problems.clear()
        _no_temperature.clear()


def _log_once(key: tuple[str, str]) -> bool:
    with _state_lock:
        if key in _reported_problems:
            return False
        _reported_problems.add(key)
        return True


# -- temperature ----------------------------------------------------------------------

_TEMPERATURE_REJECTION = re.compile(
    r"temperature[^\n]{0,80}(deprecated|not support|unsupported|not allowed)"
    r"|(unsupported|unknown)[^\n]{0,40}temperature",
    re.IGNORECASE,
)

#: How long a model's `temperature` rejection is remembered.
TEMPERATURE_MEMORY_SECONDS: Final[float] = 3600.0


def rejects_temperature(text: str) -> bool:
    """Whether an error message refuses the ``temperature`` parameter."""
    return bool(_TEMPERATURE_REJECTION.search(text))


def _remember_no_temperature(key: tuple[str, str]) -> None:
    with _state_lock:
        _no_temperature[key] = clock() + TEMPERATURE_MEMORY_SECONDS


def _drops_temperature(key: tuple[str, str]) -> bool:
    with _state_lock:
        expiry = _no_temperature.get(key)
        if expiry is None:
            return False
        if expiry < clock():
            del _no_temperature[key]
            return False
        return True


_GEMINI_MODEL_PATH = re.compile(r"/models/([^/:]+):")


def _json_body(request: httpx.Request) -> dict[str, Any] | None:
    if "json" not in request.headers.get("content-type", ""):
        return None
    try:
        body = json.loads(request.content)
    except (ValueError, httpx.RequestNotRead):
        return None
    return body if isinstance(body, dict) else None


def _model_key(request: httpx.Request, body: dict[str, Any]) -> tuple[str, str] | None:
    model = body.get("model")
    if not isinstance(model, str):
        match = _GEMINI_MODEL_PATH.search(request.url.path)
        model = match.group(1) if match else None
    return (request.url.host, model) if model else None


def _without_temperature(body: dict[str, Any]) -> dict[str, Any] | None:
    """``body`` minus temperature (OpenAI/Anthropic top level, Gemini config)."""
    if "temperature" in body:
        return {key: value for key, value in body.items() if key != "temperature"}
    config = body.get("generationConfig")
    if isinstance(config, dict) and "temperature" in config:
        trimmed = {key: value for key, value in config.items() if key != "temperature"}
        return {**body, "generationConfig": trimmed}
    return None


def _with_body(request: httpx.Request, body: dict[str, Any]) -> httpx.Request:
    headers = [
        (name, value)
        for name, value in request.headers.multi_items()
        if name.lower() != "content-length"
    ]
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        content=json.dumps(body).encode(),
        extensions=request.extensions,
    )


def _response_with_content(
    response: httpx.Response, content: bytes, request: httpx.Request
) -> httpx.Response:
    headers = [
        (name, value)
        for name, value in response.headers.multi_items()
        if name.lower() not in ("content-length", "content-encoding")
    ]
    return httpx.Response(
        response.status_code,
        headers=headers,
        content=content,
        request=request,
        extensions=response.extensions,
    )


def _hint_if_not_a_model_service(content: bytes, request: httpx.Request) -> None:
    """Point at model-service names when a name isn't one (logged once per name)."""
    if b"Unity Catalog model services" not in content:
        return
    if _log_once((request.url.host, f"not-a-model-service:{request.url.path}")):
        logger.warning(
            "Databricks Unity Gateway serves Unity Catalog model services: use a "
            "model service name such as 'system.ai.<model>' or "
            "'<catalog>.<schema>.<service>' (see %s). To keep calling the model "
            "through LiteLLM instead, pass is_litellm=True.",
            DOCS_URL,
        )


def _finish_error(
    response: httpx.Response, content: bytes, request: httpx.Request
) -> httpx.Response:
    # Inside a transport the response has no request attached yet, so the one
    # that was sent is passed in.
    _hint_if_not_a_model_service(content, request)
    try:
        body = json.loads(content)
    except ValueError:
        return _response_with_content(response, content, request)
    inner = unwrap_gateway_error_body(body)
    if inner is not body:
        content = json.dumps(inner).encode()
    return _response_with_content(response, content, request)


class _Plan:
    """What to send for one request, and the fallback if `temperature` is refused."""

    def __init__(self, request: httpx.Request) -> None:
        self.request = request
        self.retry: httpx.Request | None = None
        self.key: tuple[str, str] | None = None
        body = _json_body(request)
        if body is None:
            return
        self.key = _model_key(request, body)
        trimmed = _without_temperature(body)
        if trimmed is None or self.key is None:
            return
        if _drops_temperature(self.key):
            self.request = _with_body(request, trimmed)
        else:
            self.retry = _with_body(request, trimmed)

    def retry_after(
        self, response: httpx.Response, content: bytes
    ) -> httpx.Request | None:
        if (
            self.retry is None
            or self.key is None
            or response.status_code not in (400, 422)
        ):
            return None
        if not rejects_temperature(content.decode("utf-8", errors="replace")):
            return None
        _remember_no_temperature(self.key)
        if _log_once((self.key[0], f"temperature:{self.key[1]}")):
            logger.warning(
                "Model service %r rejects `temperature`; crewAI resends the request "
                "without it and leaves it out for this model for the next hour.",
                self.key[1],
            )
        return self.retry


# -- proxies ------------------------------------------------------------------------------


def _bypasses_proxy(url: httpx.URL, no_proxy: str) -> bool:
    """Whether ``NO_PROXY`` covers ``url``, by httpx's rules.

    ``*`` covers everything; ``scheme://host`` entries match that scheme and
    host; IPs and ``localhost`` match exactly; any other entry matches hosts
    that end with it (``example.com`` and ``.example.com`` both cover
    ``api.example.com``).
    """
    host = url.host.lower()
    for raw in no_proxy.split(","):
        entry = raw.strip().lower()
        if not entry:
            continue
        if entry == "*":
            return True
        if "://" in entry:
            scheme, _, rest = entry.partition("://")
            if url.scheme == scheme and host == rest.split("/", 1)[0].split(":", 1)[0]:
                return True
            continue
        name = entry.strip("[]")
        if _is_ip_or_localhost(name):
            if host == name:
                return True
            continue
        if host.endswith(name.lstrip("*")):
            return True
    return False


def _is_ip_or_localhost(name: str) -> bool:
    if name == "localhost":
        return True
    try:
        ipaddress.ip_address(name.split("/", 1)[0])
    except ValueError:
        return False
    return True


def _env_proxy_for(url: httpx.URL, proxies: dict[str, str]) -> str | None:
    """The proxy httpx's default client would use for ``url``.

    ``proxies`` comes from the same source as httpx:
    ``urllib.request.getproxies()``, which reads the ``*_PROXY`` environment
    variables and, when none are set, the macOS or Windows system proxy
    settings. Like httpx, it is read once, when the client is built.
    """
    proxy = proxies.get(url.scheme) or proxies.get("all")
    if not proxy or _bypasses_proxy(url, proxies.get("no", "")):
        return None
    return proxy if "://" in proxy else f"http://{proxy}"


class _EnvProxyTransport(httpx.BaseTransport):
    """Direct transport that routes through env-configured proxies per request.

    A client built with an explicit ``transport=`` no longer reads proxy
    environment variables, so this restores httpx's default behaviour.
    """

    def __init__(self) -> None:
        self._direct = httpx.HTTPTransport()
        self._proxies = urllib.request.getproxies()
        self._proxied: dict[str, httpx.HTTPTransport] = {}
        self._lock = threading.Lock()

    def _for(self, url: httpx.URL) -> httpx.BaseTransport:
        proxy = _env_proxy_for(url, self._proxies)
        if proxy is None:
            return self._direct
        with self._lock:
            transport = self._proxied.get(proxy)
            if transport is None:
                transport = self._proxied[proxy] = httpx.HTTPTransport(proxy=proxy)
        return transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._for(request.url).handle_request(request)

    def close(self) -> None:
        self._direct.close()
        for transport in self._proxied.values():
            transport.close()


class _AsyncEnvProxyTransport(httpx.AsyncBaseTransport):
    """Asynchronous counterpart of :class:`_EnvProxyTransport`."""

    def __init__(self) -> None:
        self._direct = httpx.AsyncHTTPTransport()
        self._proxies = urllib.request.getproxies()
        self._proxied: dict[str, httpx.AsyncHTTPTransport] = {}
        self._lock = threading.Lock()

    def _for(self, url: httpx.URL) -> httpx.AsyncBaseTransport:
        proxy = _env_proxy_for(url, self._proxies)
        if proxy is None:
            return self._direct
        with self._lock:
            transport = self._proxied.get(proxy)
            if transport is None:
                transport = self._proxied[proxy] = httpx.AsyncHTTPTransport(proxy=proxy)
        return transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._for(request.url).handle_async_request(request)

    async def aclose(self) -> None:
        await self._direct.aclose()
        for transport in self._proxied.values():
            await transport.aclose()


# -- transports ---------------------------------------------------------------------------


class GatewayTransport(httpx.BaseTransport):
    """Synchronous transport for Databricks Unity Gateway clients (see module docs)."""

    def __init__(self, inner: httpx.BaseTransport | None = None) -> None:
        self._inner = inner or _EnvProxyTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        plan = _Plan(request)
        sent = plan.request
        response = self._inner.handle_request(sent)
        if response.status_code < 400:
            return response
        content = response.read()
        response.close()
        retry = plan.retry_after(response, content)
        if retry is not None:
            sent = retry
            response = self._inner.handle_request(sent)
            if response.status_code < 400:
                return response
            content = response.read()
            response.close()
        return _finish_error(response, content, sent)

    def close(self) -> None:
        self._inner.close()


class AsyncGatewayTransport(httpx.AsyncBaseTransport):
    """Asynchronous counterpart of :class:`GatewayTransport`."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or _AsyncEnvProxyTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        plan = _Plan(request)
        sent = plan.request
        response = await self._inner.handle_async_request(sent)
        if response.status_code < 400:
            return response
        content = await response.aread()
        await response.aclose()
        retry = plan.retry_after(response, content)
        if retry is not None:
            sent = retry
            response = await self._inner.handle_async_request(sent)
            if response.status_code < 400:
                return response
            content = await response.aread()
            await response.aclose()
        return _finish_error(response, content, sent)

    async def aclose(self) -> None:
        await self._inner.aclose()


# -- clients ------------------------------------------------------------------------------


def _is_token_count(path: str) -> bool:
    # Token counting is an optional endpoint the SDKs fall back from on any
    # error; a 404 there says nothing about the model service or grants.
    return "count_tokens" in path or ":countTokens" in path


def _report_response_problem(auth: DatabricksAuth, response: httpx.Response) -> None:
    """Explain failures whose raw SDK errors are misleading (logged once)."""
    status = response.status_code
    host = response.request.url.host
    if 300 <= status < 400:
        if _log_once((host, "redirect")):
            logger.error(
                "Databricks returned HTTP %s redirecting to %r for %s. The "
                "credentials (%s) are most likely not valid for this workspace "
                "host, or the host is not a Databricks workspace URL.",
                status,
                response.headers.get("location", ""),
                host,
                auth.describe(),
            )
    elif (
        status == 404
        and "/ai-gateway/" in response.request.url.path
        and not _is_token_count(response.request.url.path)
    ):
        if _log_once((host, response.request.url.path)):
            logger.warning(
                "Databricks Unity Gateway returned 404 for %s. Check the model "
                "service name, and that the calling identity (%s) has USE CATALOG, "
                "USE SCHEMA and EXECUTE on it: Unity Catalog reports services the "
                "caller cannot access as not existing.",
                response.request.url.path,
                auth.describe(),
            )


def build_httpx_client(
    auth: DatabricksAuth,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: httpx.Timeout | float | None = None,
) -> httpx.Client:
    """An ``httpx.Client`` for Databricks Unity Gateway calls.

    Signs each request with a fresh token and the crewAI User-Agent, and wraps
    ``transport`` (e.g. a crewAI interceptor transport; by default a direct
    transport that honours proxy environment variables) in
    :class:`GatewayTransport`. Redirects are not followed: a Databricks redirect
    means a login page, which must not reach the SDK as if it were a response.
    """

    def on_response(response: httpx.Response) -> None:
        _report_response_problem(auth, response)

    return httpx.Client(
        auth=auth.httpx_auth(),
        transport=GatewayTransport(transport),
        follow_redirects=False,
        timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT,
        event_hooks={"response": [on_response]},
    )


def build_async_httpx_client(
    auth: DatabricksAuth,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: httpx.Timeout | float | None = None,
) -> httpx.AsyncClient:
    """Async counterpart of :func:`build_httpx_client`."""

    async def on_response(response: httpx.Response) -> None:
        _report_response_problem(auth, response)

    return httpx.AsyncClient(
        auth=auth.httpx_auth(),
        transport=AsyncGatewayTransport(transport),
        follow_redirects=False,
        timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT,
        event_hooks={"response": [on_response]},
    )

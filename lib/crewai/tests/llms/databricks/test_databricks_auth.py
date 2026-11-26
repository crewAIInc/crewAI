from __future__ import annotations

import asyncio
import copy
import json
import logging
import sys
import types
from typing import Any

import httpx
import pytest

from crewai.llms.providers.databricks import gateway as gateway_module
from crewai.llms.providers.databricks.auth import (
    DatabricksAuth,
    DatabricksAuthError,
    normalize_host,
)
from crewai.llms.providers.databricks.gateway import (
    build_async_httpx_client,
    build_httpx_client,
    gateway_base_url,
    native_family,
    unwrap_gateway_error_body,
)
from crewai.llms.providers.databricks.telemetry import (
    USER_AGENT_PRODUCT,
    user_agent,
    with_user_agent,
)
from tests.llms.databricks.fakes import HOST, FakeConfig

@pytest.fixture
def recorded_config(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the SDK Config class and record the kwargs it is built with."""
    calls: list[dict[str, Any]] = []

    class RecordingConfig(FakeConfig):
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            super().__init__(host=kwargs.get("host") or HOST)

    core = types.ModuleType("databricks.sdk.core")
    core.Config = RecordingConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "databricks.sdk.core", core)
    return calls


def _client(handler: Any, auth: DatabricksAuth | None = None) -> httpx.Client:
    return build_httpx_client(
        auth or DatabricksAuth(config=FakeConfig()),
        transport=httpx.MockTransport(handler),
    )


class TestHostAndRouting:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("dbc-test.cloud.databricks.com", HOST),
            (f"{HOST}/", HOST),
            (f"{HOST}/ai-gateway/mlflow/v1", HOST),
            (f"{HOST}/serving-endpoints", HOST),
        ],
    )
    def test_normalizes_to_workspace_url(self, value: str, expected: str) -> None:
        assert normalize_host(value) == expected

    @pytest.mark.parametrize(
        "value", ["http://dbc-test.cloud.databricks.com", "ftp://dbc-test.cloud.databricks.com"]
    )
    def test_refuses_unencrypted_hosts(self, value: str) -> None:
        with pytest.raises(DatabricksAuthError, match="https"):
            normalize_host(value)

    def test_allows_http_for_a_local_test_server(self) -> None:
        assert normalize_host("http://localhost:8080/") == "http://localhost:8080"

    def test_gateway_base_urls(self) -> None:
        assert gateway_base_url(HOST, "mlflow") == f"{HOST}/ai-gateway/mlflow/v1"
        assert gateway_base_url(HOST, "anthropic") == f"{HOST}/ai-gateway/anthropic"
        assert gateway_base_url(f"{HOST}/", "gemini") == f"{HOST}/ai-gateway/gemini"

    @pytest.mark.parametrize(
        ("model", "family"),
        [
            ("system.ai.claude-sonnet-5", "anthropic"),
            ("system.ai.gemini-3-5-flash", "gemini"),
            ("system.ai.gpt-6-sol", None),
            ("system.ai.llama-4-maverick", None),
            # A custom service name says nothing about its model.
            ("main.ai.claude_like_assistant", None),
        ],
    )
    def test_native_family_comes_from_system_ai_names_only(
        self, model: str, family: str | None
    ) -> None:
        assert native_family(model) == family


class TestCredentialResolution:
    def test_explicit_token_is_used_as_a_pat_without_sdk_round_trips(
        self, recorded_config: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An explicit token wins over an ambient profile.
        monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "someone-else")

        auth = DatabricksAuth(host=HOST, token="dapi-x")

        assert auth.host == HOST
        assert auth.auth_headers() == {"Authorization": "Bearer dapi-x"}
        assert recorded_config == []

    def test_sdk_config_carries_the_crewai_product(
        self, recorded_config: list[dict[str, Any]]
    ) -> None:
        DatabricksAuth(host=HOST).auth_headers()

        (kwargs,) = recorded_config
        assert kwargs["product"] == USER_AGENT_PRODUCT
        assert kwargs["product_version"]

    def test_profile_is_passed_to_sdk(self, recorded_config: list[dict[str, Any]]) -> None:
        DatabricksAuth(profile="dev").auth_headers()

        (kwargs,) = recorded_config
        assert kwargs["profile"] == "dev"
        assert "token" not in kwargs

    def test_oauth_m2m_from_environment_is_left_to_the_sdk(
        self,
        recorded_config: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "sp-secret")

        DatabricksAuth().auth_headers()

        (kwargs,) = recorded_config
        assert set(kwargs) == {"product", "product_version"}

    def test_explicit_oauth_client_credentials(
        self, recorded_config: list[dict[str, Any]]
    ) -> None:
        DatabricksAuth(host=HOST, client_id="sp-id", client_secret="sp-secret").host

        (kwargs,) = recorded_config
        assert kwargs["client_id"] == "sp-id"
        assert kwargs["client_secret"] == "sp-secret"

    def test_construction_does_no_io(self, recorded_config: list[dict[str, Any]]) -> None:
        auth = DatabricksAuth(profile="dev")

        assert recorded_config == []
        assert auth.resolved_host is None

    def test_supplied_config_is_used_as_is(self) -> None:
        auth = DatabricksAuth(config=FakeConfig())

        assert auth.auth_headers()["Authorization"] == "Bearer tok-1"
        assert auth.auth_headers()["Authorization"] == "Bearer tok-2"

    def test_workspace_client_config_is_used(self) -> None:
        client = types.SimpleNamespace(config=FakeConfig())
        auth = DatabricksAuth(workspace_client=client)

        assert auth.host == HOST
        assert auth.auth_headers()["Authorization"] == "Bearer tok-1"

    def test_sdk_errors_do_not_leak_config_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FailingConfig:
            def __init__(self, **_: Any) -> None:
                raise ValueError(
                    "validate: more than one authorization method configured: "
                    "oauth and pat. Config: host=x, client_id=sp-id, token=***"
                )

        core = types.ModuleType("databricks.sdk.core")
        core.Config = FailingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "databricks.sdk.core", core)

        with pytest.raises(DatabricksAuthError) as error:
            DatabricksAuth().auth_headers()

        assert "more than one authorization method" in str(error.value)
        assert "sp-id" not in str(error.value)

    def test_missing_host_is_a_clear_error(self) -> None:
        with pytest.raises(DatabricksAuthError, match="host is not configured"):
            _ = DatabricksAuth(config=FakeConfig(host="")).host

    def test_copies_share_one_instance(self) -> None:
        auth = DatabricksAuth(config=FakeConfig())
        assert copy.copy(auth) is auth
        assert copy.deepcopy(auth) is auth


class TestWithoutDatabricksSdk:
    @pytest.fixture(autouse=True)
    def _hide_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A None entry makes `import databricks.sdk.core` raise ImportError.
        monkeypatch.setitem(sys.modules, "databricks.sdk.core", None)

    def test_pat_from_environment_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", "dbc-test.cloud.databricks.com")
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")

        auth = DatabricksAuth()

        assert auth.host == HOST
        assert auth.auth_headers() == {"Authorization": "Bearer dapi-x"}

    def test_oauth_requires_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", HOST)
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-id")

        with pytest.raises(DatabricksAuthError, match=r"crewai\[databricks\]"):
            DatabricksAuth().auth_headers()


class TestUserAgentScope:
    def test_format_is_isv_underscore_product(self) -> None:
        name, _, version = user_agent().partition("/")
        assert name == "crewai_oss"
        assert version

    def test_prefix_is_added_once(self) -> None:
        once = with_user_agent("Anthropic/Python 0.73.0")
        assert once == f"{user_agent()} Anthropic/Python 0.73.0"
        assert with_user_agent(once) == once
        assert with_user_agent(None) == user_agent()

    def test_sdk_global_user_agent_is_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        useragent = pytest.importorskip("databricks.sdk.useragent")
        from databricks.sdk import config as sdk_config

        # Keep the real SDK Config offline.
        def no_metadata(*_: Any, **__: Any) -> Any:
            raise ValueError("offline")

        monkeypatch.setattr(sdk_config, "get_host_metadata", no_metadata, raising=False)
        # A token from the environment with an explicit auth type goes through
        # the provider's own SDK Config (not the static-token shortcut).
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-x")
        before = useragent.to_string()

        auth = DatabricksAuth(host=HOST, auth_type="pat")
        assert auth.auth_headers() == {"Authorization": "Bearer dapi-x"}

        assert not auth.is_static
        assert auth._config.user_agent.startswith(f"{USER_AGENT_PRODUCT}/")
        assert useragent.to_string() == before
        assert USER_AGENT_PRODUCT not in useragent.to_string()


class TestHttpxClient:
    def test_each_request_gets_a_fresh_token_and_crewai_user_agent(self) -> None:
        seen: list[httpx.Request] = []
        client = _client(lambda request: seen.append(request) or httpx.Response(200, json={}))

        client.get(f"{HOST}/x", headers={"x-api-key": "vendor", "User-Agent": "OpenAI/Python 2.0"})
        client.get(f"{HOST}/x", headers={"x-goog-api-key": "vendor"})

        assert [r.headers["Authorization"] for r in seen] == ["Bearer tok-1", "Bearer tok-2"]
        assert "x-api-key" not in seen[0].headers
        assert "x-goog-api-key" not in seen[1].headers
        assert seen[0].headers["User-Agent"] == f"{user_agent()} OpenAI/Python 2.0"

    def test_async_client_signs_requests(self) -> None:
        seen: list[httpx.Request] = []

        async def run() -> None:
            async with build_async_httpx_client(
                DatabricksAuth(config=FakeConfig()),
                transport=httpx.MockTransport(
                    lambda request: seen.append(request) or httpx.Response(200, json={})
                ),
            ) as client:
                await client.get(f"{HOST}/x")

        asyncio.run(run())

        assert seen[0].headers["Authorization"] == "Bearer tok-1"
        assert seen[0].headers["User-Agent"].startswith(f"{USER_AGENT_PRODUCT}/")

    def test_login_redirects_are_not_followed(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _client(lambda request: httpx.Response(303, headers={"location": "/login"}))

        with caplog.at_level(logging.ERROR):
            response = client.post(f"{HOST}/ai-gateway/mlflow/v1/responses")

        assert response.status_code == 303
        assert "not valid for this workspace host" in caplog.text


class TestGatewayTransport:
    def test_unwraps_the_gateway_error_envelope(self) -> None:
        upstream = {"error": {"param": "reasoning_effort", "message": "nope"}}
        client = _client(
            lambda request: httpx.Response(
                400, json={"error_code": "BAD_REQUEST", "message": json.dumps(upstream)}
            )
        )

        response = client.post(f"{HOST}/ai-gateway/mlflow/v1/chat/completions", json={})

        assert response.status_code == 400
        assert response.json() == upstream

    def test_leaves_plain_errors_alone(self) -> None:
        body = {"error_code": "NOT_FOUND", "message": "'x' does not exist."}
        client = _client(lambda request: httpx.Response(404, json=body))

        assert client.post(f"{HOST}/ai-gateway/mlflow/v1/responses", json={}).json() == body

    @pytest.mark.parametrize(
        ("path", "body"),
        [
            ("/ai-gateway/anthropic/v1/messages", {"model": "m", "temperature": 0.2}),
            (
                "/ai-gateway/gemini/v1beta/models/m:generateContent",
                {"generationConfig": {"temperature": 0.2, "maxOutputTokens": 9}},
            ),
        ],
    )
    def test_retries_once_without_a_rejected_temperature(
        self, path: str, body: dict[str, Any]
    ) -> None:
        sent: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            sent.append(payload)
            if "temperature" in json.dumps(payload):
                return httpx.Response(
                    400,
                    json={
                        "error_code": "BAD_REQUEST",
                        "message": '{"message":"`temperature` is deprecated for this model."}',
                    },
                )
            return httpx.Response(200, json={"ok": True})

        client = _client(handler)

        assert client.post(f"{HOST}{path}", json=body).json() == {"ok": True}
        assert client.post(f"{HOST}{path}", json=body).json() == {"ok": True}

        # One refusal, then the parameter is left out up front.
        assert len(sent) == 3
        assert all("temperature" not in json.dumps(payload) for payload in sent[1:])
        if "generationConfig" in body:
            assert sent[-1]["generationConfig"] == {"maxOutputTokens": 9}

    def test_forgets_a_rejection_after_its_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = [1000.0]
        monkeypatch.setattr(gateway_module, "clock", lambda: now[0])
        key = ("dbc-test.cloud.databricks.com", "m")

        gateway_module._remember_no_temperature(key)
        assert gateway_module._drops_temperature(key)

        now[0] += gateway_module.TEMPERATURE_MEMORY_SECONDS + 1
        assert not gateway_module._drops_temperature(key)

    def test_other_errors_are_not_retried(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(400, json={"error_code": "BAD_REQUEST", "message": "too long"})

        client = _client(handler)
        client.post(f"{HOST}/ai-gateway/anthropic/v1/messages", json={"model": "m", "temperature": 1})

        assert len(calls) == 1

    def test_unwrap_helper(self) -> None:
        body = {"error_code": "BAD_REQUEST", "message": '{"error": {"message": "x"}}'}
        assert unwrap_gateway_error_body(body) == {"error": {"message": "x"}}
        plain = {"message": "plain"}
        assert unwrap_gateway_error_body(plain) is plain


class TestEnvironmentProxies:
    @pytest.fixture
    def built(self, monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
        """Proxies the default inner transport builds (None = direct)."""
        made: list[str | None] = []

        def fake_transport(cls: type) -> type:
            class Recording(cls):  # type: ignore[misc, valid-type]
                def __init__(self, *args: Any, proxy: Any = None, **kwargs: Any) -> None:
                    made.append(str(proxy) if proxy else None)
                    super().__init__(*args, proxy=proxy, **kwargs)

                def handle_request(self, request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json={}, request=request)

                async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                    return httpx.Response(200, json={}, request=request)

            return Recording

        monkeypatch.setattr(gateway_module.httpx, "HTTPTransport", fake_transport(httpx.HTTPTransport))
        monkeypatch.setattr(
            gateway_module.httpx, "AsyncHTTPTransport", fake_transport(httpx.AsyncHTTPTransport)
        )
        return made

    def test_https_proxy_is_used(self, monkeypatch: pytest.MonkeyPatch, built: list[str | None]) -> None:
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        client = build_httpx_client(DatabricksAuth(config=FakeConfig()))

        client.get(f"{HOST}/ai-gateway/mlflow/v1/responses")

        assert "http://proxy.internal:3128" in built

    def test_no_proxy_hosts_go_direct(self, monkeypatch: pytest.MonkeyPatch, built: list[str | None]) -> None:
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        monkeypatch.setenv("NO_PROXY", ".cloud.databricks.com")
        client = build_httpx_client(DatabricksAuth(config=FakeConfig()))

        client.get(f"{HOST}/ai-gateway/mlflow/v1/responses")

        assert "http://proxy.internal:3128" not in built

    def test_async_client_uses_the_proxy(
        self, monkeypatch: pytest.MonkeyPatch, built: list[str | None]
    ) -> None:
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")

        async def run() -> None:
            async with build_async_httpx_client(DatabricksAuth(config=FakeConfig())) as client:
                await client.get(f"{HOST}/ai-gateway/anthropic/v1/messages")

        asyncio.run(run())

        assert "http://proxy.internal:3128" in built


def test_points_at_model_service_names(caplog: pytest.LogCaptureFixture) -> None:
    client = _client(
        lambda request: httpx.Response(
            400,
            json={
                "error_code": "BAD_REQUEST",
                "message": "'databricks-x' is no longer available. Use Unity Catalog model services.",
            },
        )
    )

    with caplog.at_level(logging.WARNING):
        client.post(f"{HOST}/ai-gateway/mlflow/v1/responses", json={"model": "databricks-x"})

    assert "model service name" in caplog.text
    assert "is_litellm=True" in caplog.text


class TestProxyRules:
    @pytest.fixture
    def proxies(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
        """Stand in for urllib.request.getproxies (env vars or system settings)."""
        values: dict[str, str] = {}
        monkeypatch.setattr(gateway_module.urllib.request, "getproxies", lambda: values)
        return values

    def test_system_proxy_settings_are_used(self, proxies: dict[str, str]) -> None:
        proxies["https"] = "proxy.corp:8080"  # what macOS/Windows settings yield

        assert (
            gateway_module._env_proxy_for(httpx.URL(f"{HOST}/x"), proxies)
            == "http://proxy.corp:8080"
        )

    def test_proxy_settings_are_read_once_per_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[int] = []

        def getproxies() -> dict[str, str]:
            reads.append(1)
            return {}

        monkeypatch.setattr(gateway_module.urllib.request, "getproxies", getproxies)
        transport = gateway_module._EnvProxyTransport()
        url = httpx.URL(f"{HOST}/x")

        transport._for(url)
        transport._for(url)

        assert len(reads) == 1

    @pytest.mark.parametrize(
        ("no_proxy", "bypassed"),
        [
            ("*", True),
            ("cloud.databricks.com", True),
            (".cloud.databricks.com", True),
            ("databricks.com", True),
            ("https://dbc-test.cloud.databricks.com", True),
            ("http://dbc-test.cloud.databricks.com", False),
            ("other.example.com", False),
            ("10.0.0.1", False),
        ],
    )
    def test_no_proxy_follows_httpx_rules(
        self, proxies: dict[str, str], no_proxy: str, bypassed: bool
    ) -> None:
        proxies.update({"https": "http://proxy.corp:8080", "no": no_proxy})

        proxy = gateway_module._env_proxy_for(httpx.URL(f"{HOST}/x"), proxies)

        assert (proxy is None) is bypassed


def test_no_404_hint_for_token_counting(caplog: pytest.LogCaptureFixture) -> None:
    client = _client(lambda request: httpx.Response(404, json={"error_code": "NOT_FOUND"}))

    with caplog.at_level(logging.WARNING):
        client.post(f"{HOST}/ai-gateway/anthropic/v1/messages/count_tokens", json={})

    assert "returned 404" not in caplog.text


def test_static_tokens_sign_async_requests_without_a_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio as aio

    def no_threads(*_: Any, **__: Any) -> Any:
        raise AssertionError("no worker thread is needed for a static token")

    auth = DatabricksAuth(host=HOST, token="dapi-x")
    auth.host  # resolve
    monkeypatch.setattr(aio, "to_thread", no_threads)
    seen: list[httpx.Request] = []

    async def run() -> None:
        async with build_async_httpx_client(
            auth,
            transport=httpx.MockTransport(
                lambda request: seen.append(request) or httpx.Response(200, json={})
            ),
        ) as client:
            await client.get(f"{HOST}/x")

    aio.run(run())

    assert seen[0].headers["Authorization"] == "Bearer dapi-x"

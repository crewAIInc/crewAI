from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from crewai.llm import LLM
from crewai.llms.providers.databricks import anthropic_completion, completion, gemini_completion
from crewai.llms.providers.databricks import gateway as gateway_module
from crewai.llms.providers.databricks.anthropic_completion import (
    DatabricksAnthropicCompletion,
)
from crewai.llms.providers.databricks.auth import DatabricksAuth
from crewai.llms.providers.databricks.completion import DatabricksCompletion
from crewai.llms.providers.databricks.gemini_completion import (
    DatabricksGeminiCompletion,
)
from crewai.llms.providers.databricks.telemetry import USER_AGENT_PRODUCT
from tests.llms.databricks.fakes import HOST, FakeConfig

Handler = Callable[[httpx.Request], httpx.Response]


class Gateway:
    """Records requests sent to the (mocked) Databricks Unity Gateway."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.handler: Handler | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        return _default_response(request)

    @property
    def last_body(self) -> dict[str, Any]:
        return json.loads(self.requests[-1].content)


def _default_response(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/responses"):
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "m",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            },
        )
    if path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
    if path.endswith("/v1/messages"):
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )
    if ":generateContent" in path:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "OK"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 3,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 4,
                },
            },
        )
    return httpx.Response(404, json={"error_code": "NOT_FOUND"})


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Gateway:
    """Route every Databricks gateway client through a mock transport."""
    recorder = Gateway()
    mock = httpx.MockTransport(recorder)

    def sync_client(auth: DatabricksAuth, **_: Any) -> httpx.Client:
        return gateway_module.build_httpx_client(auth, transport=mock)

    def async_client(auth: DatabricksAuth, **_: Any) -> httpx.AsyncClient:
        return gateway_module.build_async_httpx_client(auth, transport=mock)

    for module in (completion, anthropic_completion, gemini_completion):
        monkeypatch.setattr(module, "build_httpx_client", sync_client)
        monkeypatch.setattr(module, "build_async_httpx_client", async_client)
    return recorder


def _auth() -> DatabricksAuth:
    return DatabricksAuth(config=FakeConfig())


class TestRouting:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("system.ai.claude-sonnet-5", DatabricksAnthropicCompletion),
            ("system.ai.gemini-3-5-flash", DatabricksGeminiCompletion),
            ("system.ai.gpt-6-sol", DatabricksCompletion),
            ("system.ai.llama-4-maverick", DatabricksCompletion),
            ("main.ai.support_assistant", DatabricksCompletion),
        ],
    )
    def test_model_service_name_picks_the_gateway_api(
        self, model: str, expected: type
    ) -> None:
        llm = LLM(model=f"databricks/{model}")

        assert type(llm) is expected
        assert llm.provider == "databricks"
        assert llm.model == model

    def test_api_argument_picks_a_native_api_for_custom_services(self) -> None:
        assert isinstance(
            LLM(model="databricks/main.ai.my_claude", api="anthropic"),
            DatabricksAnthropicCompletion,
        )
        assert isinstance(
            LLM(model="databricks/main.ai.my_gemini", api="gemini"),
            DatabricksGeminiCompletion,
        )

    def test_api_argument_can_force_the_unified_apis(self) -> None:
        llm = LLM(model="databricks/system.ai.claude-sonnet-5", api="completions")

        assert isinstance(llm, DatabricksCompletion)
        assert llm.api == "completions"

    def test_missing_native_sdk_falls_back_to_unified_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(completion, "_native_class", lambda family: None)

        llm = LLM(model="databricks/system.ai.claude-haiku-4-5")

        assert isinstance(llm, DatabricksCompletion)
        assert llm.api == "responses"

    def test_litellm_route_is_still_available(self) -> None:
        pytest.importorskip("litellm")

        llm = LLM(model="databricks/system.ai.gpt-6-sol", is_litellm=True)

        assert not isinstance(llm, DatabricksCompletion)
        assert llm.is_litellm

    def test_rejects_unsupported_api(self) -> None:
        with pytest.raises(ValueError, match="Unsupported Databricks api"):
            DatabricksCompletion(model="system.ai.gpt-6-sol", api="native")

    @pytest.mark.parametrize("option", ["auto_chain", "previous_response_id", "store"])
    def test_rejects_stored_response_features(self, option: str) -> None:
        value: Any = "resp_1" if option == "previous_response_id" else True
        with pytest.raises(ValueError, match="not supported on Databricks"):
            DatabricksCompletion(model="system.ai.gpt-6-sol", **{option: value})


class TestConstruction:
    @pytest.fixture
    def sdk_configs(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []

        class RecordingConfig(FakeConfig):
            def __init__(self, **kwargs: Any) -> None:
                built.append(kwargs)
                super().__init__()

        core = types.ModuleType("databricks.sdk.core")
        core.Config = RecordingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "databricks.sdk.core", core)
        return built

    @pytest.mark.parametrize(
        "model",
        ["system.ai.claude-sonnet-5", "system.ai.gemini-3-5-flash", "system.ai.gpt-6-sol"],
    )
    def test_building_an_llm_does_no_io(
        self, model: str, sdk_configs: list[dict[str, Any]], gateway: Gateway
    ) -> None:
        llm = LLM(model=f"databricks/{model}", profile="dev")

        assert sdk_configs == []
        assert gateway.requests == []

        llm.call("hi")

        assert len(sdk_configs) == 1
        assert sdk_configs[0]["profile"] == "dev"

    def test_vendor_keys_are_never_used(self, gateway: Gateway) -> None:
        for model in ("system.ai.claude-sonnet-5", "system.ai.gemini-3-5-flash", "system.ai.gpt-6-sol"):
            llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())
            assert llm.api_key is None
            llm.call("hi")

        for request in gateway.requests:
            assert "must-not-be-used" not in str(request.headers)
            assert request.headers["Authorization"].startswith("Bearer tok-")

    def test_api_key_is_a_databricks_pat(self, gateway: Gateway) -> None:
        llm = LLM(
            model="databricks/system.ai.gemini-3-5-flash",
            api_key="dapi-secret",
            host=HOST,
        )

        llm.call("hi")

        assert gateway.requests[0].headers["Authorization"] == "Bearer dapi-secret"
        assert "dapi-secret" not in repr(llm)


class TestGatewayApis:
    def test_responses_call_uses_gateway_with_fresh_tokens(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/system.ai.gpt-6-sol", databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert llm.call("hi again") == "OK"

        first, second = gateway.requests
        assert first.url == f"{HOST}/ai-gateway/mlflow/v1/responses"
        assert first.headers["Authorization"] == "Bearer tok-1"
        assert second.headers["Authorization"] == "Bearer tok-2"
        assert first.headers["User-Agent"].startswith(f"{USER_AGENT_PRODUCT}/")
        assert gateway.last_body["model"] == "system.ai.gpt-6-sol"

    def test_chat_completions_api(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/main.ai.svc", api="completions", databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert gateway.requests[0].url == f"{HOST}/ai-gateway/mlflow/v1/chat/completions"

    def test_explicit_base_url_is_honoured(self, gateway: Gateway) -> None:
        llm = LLM(
            model="databricks/main.ai.svc",
            base_url="https://gateway.example.com/ai-gateway/mlflow/v1",
            databricks_auth=_auth(),
        )

        llm.call("hi")

        assert gateway.requests[0].url.host == "gateway.example.com"

    def test_async_call(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/system.ai.gpt-6-sol", databricks_auth=_auth())

        async def run() -> Any:
            return await llm.acall("hi")

        assert asyncio.run(run()) == "OK"
        assert gateway.requests[0].headers["Authorization"] == "Bearer tok-1"

    def test_anthropic_messages_on_gateway(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/system.ai.claude-haiku-4-5", databricks_auth=_auth())

        assert llm.call("hi") == "OK"

        request = gateway.requests[0]
        assert request.url == f"{HOST}/ai-gateway/anthropic/v1/messages"
        assert request.headers["Authorization"] == "Bearer tok-1"
        assert "x-api-key" not in request.headers
        assert request.headers["User-Agent"].startswith(f"{USER_AGENT_PRODUCT}/")

    def test_gemini_generate_content_on_gateway(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/main.ai.gemini_service", api="gemini", databricks_auth=_auth())

        assert llm.call("hi") == "OK"

        request = gateway.requests[0]
        assert request.url.path == (
            "/ai-gateway/gemini/v1beta/models/main.ai.gemini_service:generateContent"
        )
        assert request.headers["Authorization"] == "Bearer tok-1"
        assert "x-goog-api-key" not in request.headers
        assert request.headers["User-Agent"].startswith(f"{USER_AGENT_PRODUCT}/")

    def test_gemini_honours_max_tokens(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/system.ai.gemini-3-5-flash", max_tokens=777, databricks_auth=_auth())

        llm.call("hi")

        assert gateway.last_body["generationConfig"]["maxOutputTokens"] == 777

    def test_gemini_tools_do_not_depend_on_the_service_name(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/main.ai.gemini_service", api="gemini", databricks_auth=_auth())

        assert llm.supports_function_calling()
        llm.call("Weather in Paris?", tools=[WEATHER_TOOL])

        declarations = [
            declaration["name"]
            for tool in gateway.last_body["tools"]
            for declaration in tool["functionDeclarations"]
        ]
        assert declarations == ["get_weather"]


class TestGatewayBehaviour:
    def test_temperature_rejection_is_retried_inside_one_call(
        self, gateway: Gateway, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "temperature" in json.loads(request.content):
                return httpx.Response(
                    400,
                    json={
                        "error_code": "BAD_REQUEST",
                        "message": '{"message":"`temperature` is deprecated for this model."}',
                    },
                )
            return _default_response(request)

        gateway.handler = handler
        llm = LLM(
            model="databricks/system.ai.claude-sonnet-5",
            temperature=0.2,
            databricks_auth=_auth(),
        )

        with caplog.at_level(logging.ERROR):
            assert llm.call("hi") == "OK"
            assert llm.call("again") == "OK"

        bodies = [json.loads(request.content) for request in gateway.requests]
        assert "temperature" in bodies[0]
        assert all("temperature" not in body for body in bodies[1:])
        assert len(gateway.requests) == 3  # one refusal, then omitted up front
        assert "API call failed" not in caplog.text  # no failed call is reported

    def test_errors_reach_the_sdk_in_the_upstream_shape(self, gateway: Gateway) -> None:
        import openai

        from crewai.llms.providers.openai.completion import OpenAICompletion

        upstream = {
            "error": {
                "message": "Function tools with reasoning_effort are not supported for "
                "gpt-6-sol in /v1/chat/completions. To use function tools, use "
                "/v1/responses or set reasoning_effort to 'none'.",
                "param": "reasoning_effort",
            }
        }
        gateway.handler = lambda request: httpx.Response(
            400, json={"error_code": "BAD_REQUEST", "message": json.dumps(upstream)}
        )
        llm = LLM(model="databricks/system.ai.gpt-6-sol", api="completions", databricks_auth=_auth())

        with pytest.raises(openai.BadRequestError) as error:
            llm._get_sync_client().chat.completions.create(
                model=llm.model, messages=[{"role": "user", "content": "hi"}]
            )

        # crewAI's own OpenAI recovery recognises the unwrapped error.
        assert OpenAICompletion._rejects_reasoning_effort_with_tools(error.value)

    @pytest.mark.parametrize(
        "model", ["openai/gpt-4o", "anthropic/claude-sonnet-4-5"]
    )
    def test_other_providers_never_send_the_databricks_user_agent(
        self, monkeypatch: pytest.MonkeyPatch, gateway: Gateway, model: str
    ) -> None:
        # A Databricks LLM has run first in this process.
        LLM(model="databricks/system.ai.gpt-6-sol", databricks_auth=_auth()).call("hi")
        assert USER_AGENT_PRODUCT in gateway.requests[0].headers["User-Agent"]

        seen: list[httpx.Request] = []

        def socket(self: Any, request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _default_response(request)

        # The vendor provider builds its own default client; only the socket
        # layer is mocked.
        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", socket)
        plain = LLM(model=model, api_key="vendor-test-key")

        plain.call("hi")

        assert seen, "the vendor provider made no request"
        assert USER_AGENT_PRODUCT not in seen[0].headers.get("User-Agent", "")


class TestConfigAndCapabilities:
    def test_to_config_dict_round_trips_without_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = LLM(
            model="databricks/system.ai.gpt-6-sol",
            api_key="dapi-secret",
            host=HOST,
        )

        config = llm.to_config_dict()

        assert config["model"] == "databricks/system.ai.gpt-6-sol"
        assert config["api"] == "responses"
        assert config["host"] == HOST
        assert "dapi-secret" not in json.dumps(config, default=str)
        # Rebuilt elsewhere, credentials come from that environment.
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-other")
        rebuilt = LLM(**config)
        assert isinstance(rebuilt, DatabricksCompletion)
        assert rebuilt._databricks_auth().auth_headers() == {
            "Authorization": "Bearer dapi-other"
        }

    def test_native_classes_round_trip(self) -> None:
        for model, expected in (
            ("main.ai.my_claude", DatabricksAnthropicCompletion),
            ("main.ai.my_gemini", DatabricksGeminiCompletion),
        ):
            api = "anthropic" if expected is DatabricksAnthropicCompletion else "gemini"
            config = LLM(model=f"databricks/{model}", api=api, host=HOST).to_config_dict()
            assert isinstance(LLM(**config), expected)

    def test_context_window_uses_catalog_aliases(self) -> None:
        llm = LLM(model="databricks/system.ai.gpt-5-4-mini")

        names = llm._context_window_names()

        assert "gpt-5-4-mini" in names
        assert "gpt-5.4-mini" in names


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _sse(events: list[tuple[str | None, dict[str, Any] | str]]) -> httpx.Response:
    lines = []
    for event, data in events:
        if event:
            lines.append(f"event: {event}")
        lines.append(f"data: {data if isinstance(data, str) else json.dumps(data)}")
        lines.append("")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=("\n".join(lines) + "\n").encode(),
    )


class TestRoutingRules:
    @pytest.mark.parametrize(
        "model",
        [
            "databricks-meta-llama-3-3-70b-instruct",  # endpoint-style name
            "system.ai.gpt-6-sol",
            "main.ai.support_bot",
            "a.b.c.d",
        ],
    )
    def test_any_name_uses_databricks_unity_gateway(self, model: str) -> None:
        llm = LLM(model=f"databricks/{model}")

        assert llm.provider == "databricks"
        assert not getattr(llm, "is_litellm", False)

    @pytest.mark.parametrize(
        ("model", "api"),
        [("system.ai.gpt-6-sol", "anthropic"), ("system.ai.claude-sonnet-5", "gemini")],
    )
    def test_native_api_must_match_a_system_ai_model(self, model: str, api: str) -> None:
        with pytest.raises(ValueError, match="not served by the native"):
            LLM(model=f"databricks/{model}", api=api)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"api": "native"}, "Unsupported Databricks api"),
            ({"store": True}, "not supported on Databricks"),
        ],
    )
    def test_invalid_options_are_reported_through_llm(
        self, kwargs: dict[str, Any], message: str
    ) -> None:
        # LLM() reports provider construction errors as ImportError (pre-existing
        # behaviour for every native provider).
        with pytest.raises(ImportError, match=message):
            LLM(model="databricks/system.ai.gpt-6-sol", **kwargs)

    def test_no_client_is_built_until_the_first_call(self) -> None:
        for model in ("system.ai.claude-sonnet-5", "system.ai.gemini-3-5-flash", "system.ai.gpt-6-sol"):
            llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())
            assert llm._client is None
            assert getattr(llm, "_async_client", None) is None

    def test_api_base_is_an_alias_of_base_url(self, gateway: Gateway) -> None:
        llm = LLM(
            model="databricks/system.ai.claude-haiku-4-5",
            api_base="https://gateway.example.com/ai-gateway/anthropic",
            databricks_auth=_auth(),
        )

        llm.call("hi")

        assert gateway.requests[0].url.host == "gateway.example.com"


class TestStreaming:
    def test_chat_completions_stream(self, gateway: Gateway) -> None:
        def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        gateway.handler = lambda request: _sse(
            [
                (None, chunk({"role": "assistant", "content": "O"})),
                (None, chunk({"content": "K"})),
                (None, chunk({}, "stop")),
                (None, "[DONE]"),
            ]
        )
        llm = LLM(model="databricks/main.ai.svc", api="completions", stream=True, databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert gateway.last_body["stream"] is True

    def test_anthropic_stream(self, gateway: Gateway) -> None:
        gateway.handler = lambda request: _sse(
            [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "m",
                            "content": [],
                            "stop_reason": None,
                            "usage": {"input_tokens": 3, "output_tokens": 0},
                        },
                    },
                ),
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        )
        llm = LLM(model="databricks/system.ai.claude-haiku-4-5", stream=True, databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert gateway.requests[0].url.path == "/ai-gateway/anthropic/v1/messages"

    def test_gemini_stream(self, gateway: Gateway) -> None:
        def piece(text: str) -> dict[str, Any]:
            return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}

        gateway.handler = lambda request: _sse([(None, piece("O")), (None, piece("K"))])
        llm = LLM(model="databricks/system.ai.gemini-3-5-flash", stream=True, databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert ":streamGenerateContent" in gateway.requests[0].url.path


class TestToolCalling:
    def test_gemini_round_trip_keeps_the_thought_signature(self, gateway: Gateway) -> None:
        from crewai.utilities.agent_utils import build_tool_calls_assistant_message

        replies = iter(
            [
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
                                        "thoughtSignature": "c2lnbmF0dXJl",
                                    }
                                ],
                            }
                        }
                    ]
                },
                {"candidates": [{"content": {"role": "model", "parts": [{"text": "Sunny"}]}}]},
            ]
        )
        gateway.handler = lambda request: httpx.Response(200, json=next(replies))
        llm = LLM(model="databricks/system.ai.gemini-3-5-flash", databricks_auth=_auth())
        messages: list[Any] = [{"role": "user", "content": "Weather in Paris?"}]

        tool_calls = llm.call(messages, tools=[WEATHER_TOOL])
        assistant, reported = build_tool_calls_assistant_message(tool_calls)
        messages += [
            assistant,
            {"role": "tool", "tool_call_id": reported[0]["id"], "name": "get_weather", "content": "Sunny"},
        ]
        answer = llm.call(messages, tools=[WEATHER_TOOL])

        assert answer == "Sunny"
        assert "c2lnbmF0dXJl" in gateway.requests[1].content.decode()

    def test_claude_round_trip(self, gateway: Gateway) -> None:
        from crewai.utilities.agent_utils import build_tool_calls_assistant_message

        replies = iter(
            [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                },
                {
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": [{"type": "text", "text": "Sunny"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 9, "output_tokens": 1},
                },
            ]
        )
        gateway.handler = lambda request: httpx.Response(200, json=next(replies))
        llm = LLM(model="databricks/system.ai.claude-haiku-4-5", databricks_auth=_auth())
        messages: list[Any] = [{"role": "user", "content": "Weather in Paris?"}]

        tool_calls = llm.call(messages, tools=[WEATHER_TOOL])
        assistant, reported = build_tool_calls_assistant_message(tool_calls)
        messages += [
            assistant,
            {"role": "tool", "tool_call_id": reported[0]["id"], "name": "get_weather", "content": "Sunny"},
        ]

        assert llm.call(messages, tools=[WEATHER_TOOL]) == "Sunny"
        assert "toolu_1" in gateway.requests[1].content.decode()
        assert gateway.requests[0].content and "get_weather" in gateway.requests[0].content.decode()


class TestUsageAndAsync:
    @pytest.mark.parametrize(
        "model", ["system.ai.claude-haiku-4-5", "system.ai.gemini-3-5-flash", "system.ai.gpt-6-sol"]
    )
    def test_token_usage_is_tracked(self, gateway: Gateway, model: str) -> None:
        llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())

        llm.call("hi")

        assert llm.get_token_usage_summary().total_tokens == 4

    def test_async_temperature_retry(self, gateway: Gateway) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "temperature" in json.loads(request.content):
                return httpx.Response(
                    400,
                    json={
                        "error_code": "BAD_REQUEST",
                        "message": '{"error":{"message":"Unsupported parameter: \'temperature\' is not supported with this model.","param":"temperature"}}',
                    },
                )
            return _default_response(request)

        gateway.handler = handler
        llm = LLM(model="databricks/system.ai.gpt-6-sol", temperature=0.3, databricks_auth=_auth())

        async def run() -> Any:
            return await llm.acall("hi")

        assert asyncio.run(run()) == "OK"
        assert len(gateway.requests) == 2
        assert "temperature" not in json.loads(gateway.requests[1].content)


class TestOverlay:
    """crewAI's llm_overlay rebuilds a model with the declared LLM's settings."""

    def _remap(self, base: Any, model: str) -> Any:
        from crewai.utilities.llm_utils import create_llm_like

        return create_llm_like(f"databricks/{model}", base)

    @pytest.mark.parametrize(
        ("source", "target", "expected"),
        [
            ("system.ai.gpt-6-sol", "system.ai.gpt-5-4-mini", DatabricksCompletion),
            ("system.ai.claude-sonnet-5", "system.ai.gpt-6-sol", DatabricksCompletion),
            ("system.ai.gpt-6-sol", "system.ai.claude-sonnet-5", DatabricksAnthropicCompletion),
            ("system.ai.claude-sonnet-5", "system.ai.gemini-3-5-flash", DatabricksGeminiCompletion),
        ],
    )
    def test_remap_keeps_explicit_credentials(
        self, gateway: Gateway, source: str, target: str, expected: type
    ) -> None:
        base = LLM(model=f"databricks/{source}", api_key="dapi-explicit", host=HOST)

        remapped = self._remap(base, target)
        remapped.call("hi")

        assert type(remapped) is expected
        assert gateway.requests[0].url.host == "dbc-test.cloud.databricks.com"
        assert gateway.requests[0].headers["Authorization"] == "Bearer dapi-explicit"

    def test_remap_keeps_an_explicit_profile(self) -> None:
        base = LLM(model="databricks/system.ai.gpt-6-sol", profile="prod")

        remapped = self._remap(base, "system.ai.claude-haiku-4-5")

        assert remapped.profile == "prod"

    def test_remap_shares_resolved_credentials(self, gateway: Gateway) -> None:
        auth = _auth()
        base = LLM(model="databricks/system.ai.gemini-3-5-flash", databricks_auth=auth)
        base.call("hi")

        remapped = self._remap(base, "system.ai.gpt-6-sol")

        assert remapped._databricks_auth() is auth

    def test_remap_with_environment_credentials(
        self, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", HOST)
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-env")
        base = LLM(model="databricks/system.ai.claude-sonnet-5")

        remapped = self._remap(base, "system.ai.gpt-6-sol")
        remapped.call("hi")

        assert gateway.requests[0].headers["Authorization"] == "Bearer dapi-env"


class TestBaseUrl:
    def test_http_base_url_is_refused(self) -> None:
        llm = LLM(
            model="databricks/system.ai.gpt-6-sol",
            base_url="http://gateway.example.com/ai-gateway/mlflow/v1",
            databricks_auth=_auth(),
        )

        with pytest.raises(Exception, match="https"):
            llm.call("hi")

    @pytest.mark.parametrize(
        ("model", "path"),
        [
            ("system.ai.gpt-6-sol", "/ai-gateway/mlflow/v1/responses"),
            ("system.ai.claude-haiku-4-5", "/ai-gateway/anthropic/v1/messages"),
        ],
    )
    def test_host_only_base_url_gets_the_api_path(
        self, gateway: Gateway, model: str, path: str
    ) -> None:
        llm = LLM(
            model=f"databricks/{model}",
            base_url="https://other-workspace.cloud.databricks.com",
            databricks_auth=_auth(),
        )

        llm.call("hi")

        assert gateway.requests[0].url.host == "other-workspace.cloud.databricks.com"
        assert gateway.requests[0].url.path == path


class TestAsyncResolution:
    @pytest.mark.parametrize(
        "model",
        ["system.ai.claude-haiku-4-5", "system.ai.gemini-3-5-flash", "system.ai.gpt-6-sol"],
    )
    def test_credentials_resolve_off_the_event_loop(
        self, gateway: Gateway, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        import threading

        built_on: list[str] = []

        class RecordingConfig(FakeConfig):
            def __init__(self, **_: Any) -> None:
                built_on.append(threading.current_thread().name)
                super().__init__()

        core = types.ModuleType("databricks.sdk.core")
        core.Config = RecordingConfig  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "databricks.sdk.core", core)
        llm = LLM(model=f"databricks/{model}", profile="dev")

        async def run() -> tuple[Any, str]:
            return await llm.acall("hi"), threading.current_thread().name

        answer, loop_thread = asyncio.run(run())

        assert answer == "OK"
        assert built_on and loop_thread not in built_on


class TestGeminiClient:
    def test_vertex_env_does_not_redirect_gemini(
        self, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        llm = LLM(model="databricks/system.ai.gemini-3-5-flash", databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert gateway.requests[0].url.path.startswith("/ai-gateway/gemini/")


class SalesSummary(BaseModel):
    city: str
    amount_usd: int


class TestStructuredOutputAndStreaming:
    def test_responses_structured_output(self, gateway: Gateway) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            response = _default_response(request)
            body = response.json()
            body["output"][0]["content"][0]["text"] = '{"city": "Tokyo", "amount_usd": 4210}'
            return httpx.Response(200, json=body)

        gateway.handler = handler
        llm = LLM(model="databricks/system.ai.gpt-6-sol", databricks_auth=_auth())

        result = llm.call("Tokyo sales?", response_model=SalesSummary)

        assert "4210" in (result.model_dump_json() if isinstance(result, BaseModel) else str(result))

    def test_gemini_structured_output(self, gateway: Gateway) -> None:
        gateway.handler = lambda request: httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": '{"city": "Tokyo", "amount_usd": 4210}'}],
                        }
                    }
                ]
            },
        )
        llm = LLM(model="databricks/system.ai.gemini-3-5-flash", databricks_auth=_auth())

        result = llm.call("Tokyo sales?", response_model=SalesSummary)

        assert "4210" in (result.model_dump_json() if isinstance(result, BaseModel) else str(result))
        assert gateway.last_body["generationConfig"]["responseMimeType"] == "application/json"

    def test_claude_structured_output(self, gateway: Gateway) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            tool = next(
                (t["name"] for t in body.get("tools", []) if "name" in t), "structured_output"
            )
            if body.get("output_format") or body.get("output_config"):
                content = [{"type": "text", "text": '{"city": "Tokyo", "amount_usd": 4210}'}]
            else:
                content = [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": tool,
                        "input": {"city": "Tokyo", "amount_usd": 4210},
                    }
                ]
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "m",
                    "content": content,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            )

        gateway.handler = handler
        llm = LLM(model="databricks/system.ai.claude-haiku-4-5", databricks_auth=_auth())

        result = llm.call("Tokyo sales?", response_model=SalesSummary)

        assert "4210" in (result.model_dump_json() if isinstance(result, BaseModel) else str(result))

    def test_responses_stream(self, gateway: Gateway) -> None:
        final = _default_response(
            httpx.Request("POST", f"{HOST}/ai-gateway/mlflow/v1/responses")
        ).json()
        gateway.handler = lambda request: _sse(
            [
                ("response.created", {"type": "response.created", "sequence_number": 0, "response": {**final, "status": "in_progress", "output": []}}),
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": 1,
                        "item_id": "msg_1",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "OK",
                        "logprobs": [],
                    },
                ),
                ("response.completed", {"type": "response.completed", "sequence_number": 2, "response": final}),
            ]
        )
        llm = LLM(model="databricks/system.ai.gpt-6-sol", stream=True, databricks_auth=_auth())

        assert llm.call("hi") == "OK"
        assert gateway.last_body["stream"] is True

    def test_temperature_retry_on_a_streaming_request(self, gateway: Gateway) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if "temperature" in body:
                return httpx.Response(
                    400,
                    json={
                        "error_code": "BAD_REQUEST",
                        "message": '{"message":"`temperature` is deprecated for this model."}',
                    },
                )
            return _sse(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "msg_1",
                                "type": "message",
                                "role": "assistant",
                                "model": "m",
                                "content": [],
                                "stop_reason": None,
                                "usage": {"input_tokens": 3, "output_tokens": 0},
                            },
                        },
                    ),
                    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}}),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )

        gateway.handler = handler
        llm = LLM(
            model="databricks/system.ai.claude-sonnet-5",
            stream=True,
            temperature=0.2,
            databricks_auth=_auth(),
        )

        assert llm.call("hi") == "OK"
        resent = json.loads(gateway.requests[1].content)
        assert "temperature" not in resent and resent["stream"] is True

    @pytest.mark.parametrize("model", ["system.ai.claude-haiku-4-5", "system.ai.gemini-3-5-flash"])
    def test_native_async_calls(self, gateway: Gateway, model: str) -> None:
        llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())

        async def run() -> Any:
            return await llm.acall("hi")

        assert asyncio.run(run()) == "OK"


# A 1x1 PNG.
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _image_message() -> list[dict[str, Any]]:
    from crewai_files import ImageFile

    return [
        {
            "role": "user",
            "content": "What is in this image?",
            "files": {"pixel": ImageFile(source=_PNG)},
        }
    ]


def _blocks(value: Any) -> list[dict[str, Any]]:
    """Every dict nested anywhere in a request body."""
    if isinstance(value, dict):
        return [value] + [b for v in value.values() for b in _blocks(v)]
    if isinstance(value, list):
        return [b for v in value for b in _blocks(v)]
    return []


class TestFiles:
    @pytest.mark.parametrize(
        ("model", "block_key", "block_value"),
        [
            ("system.ai.claude-sonnet-5", "type", "image"),
            ("system.ai.gpt-6-sol", "type", "input_image"),
            ("system.ai.gemini-3-5-flash", "inlineData", None),
        ],
    )
    def test_images_reach_each_api_in_its_own_format(
        self, gateway: Gateway, model: str, block_key: str, block_value: str | None
    ) -> None:
        llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())

        assert llm.supports_multimodal()
        assert llm.call(_image_message()) == "OK"

        blocks = _blocks(gateway.last_body)
        assert any(
            block_key in block and (block_value is None or block[block_key] == block_value)
            for block in blocks
        ), gateway.last_body

    def test_chat_completions_images(self, gateway: Gateway) -> None:
        llm = LLM(model="databricks/system.ai.gpt-6-sol", api="completions", databricks_auth=_auth())

        llm.call(_image_message())

        assert any(block.get("type") == "image_url" for block in _blocks(gateway.last_body))

    def test_custom_services_accept_files(self) -> None:
        for api in ("anthropic", "gemini", "responses"):
            llm = LLM(model="databricks/main.ai.service", api=api)
            assert llm.supports_multimodal(), api

    def test_name_checks_apply_only_to_names_they_know(self) -> None:
        # GPT names are checked against crewAI's list; other families defer to
        # Databricks Unity Gateway.
        assert not LLM(model="databricks/system.ai.gpt-3-5-turbo").supports_multimodal()
        assert LLM(model="databricks/system.ai.llama-4-maverick").supports_multimodal()
        assert LLM(model="databricks/system.ai.qwen3-vl-235b").supports_multimodal()

    @pytest.mark.parametrize(
        ("model", "size"),
        [
            ("system.ai.claude-sonnet-5", 6 * 1024 * 1024),
            ("system.ai.gpt-6-sol", 6 * 1024 * 1024),
            ("system.ai.gemini-3-5-flash", 21 * 1024 * 1024),
        ],
    )
    def test_large_files_are_sent_inline_never_uploaded(
        self, monkeypatch: pytest.MonkeyPatch, gateway: Gateway, model: str, size: int
    ) -> None:
        """Above the vendor upload threshold, files still go to Databricks."""
        from crewai_files import PDFFile
        from crewai_files.resolution.resolver import FileResolver

        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.setenv(key, "vendor-key")

        def no_uploads(*_: Any, **__: Any) -> Any:
            raise AssertionError("a vendor Files API uploader was built")

        monkeypatch.setattr(FileResolver, "_get_uploader", no_uploads)
        pdf = b"%PDF-1.4\n" + b"0" * size + b"\n%%EOF\n"
        llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())

        llm.call(
            [
                {
                    "role": "user",
                    "content": "Summarise this document.",
                    "files": {"report": PDFFile(source=pdf)},
                }
            ]
        )

        assert len(gateway.requests) == 1
        assert gateway.requests[0].url.host == httpx.URL(HOST).host
        assert len(gateway.requests[0].content) > size  # the document, inline

    def test_a_crew_auto_injects_images_for_databricks_models(
        self, gateway: Gateway
    ) -> None:
        from crewai import Agent, Crew, Task
        from crewai_files import ImageFile

        llm = LLM(model="databricks/system.ai.claude-sonnet-5", databricks_auth=_auth())
        agent = Agent(role="Analyst", goal="Describe files", backstory="Brief.", llm=llm)
        task = Task(
            description="Describe the image.", expected_output="One sentence.", agent=agent
        )

        Crew(agents=[agent], tasks=[task]).kickoff(
            input_files={"pixel": ImageFile(source=_PNG)}
        )

        body = gateway.requests[0].content.decode()
        # task.py / crew.py treat the image as loaded into the conversation,
        # rather than handing it to the model through a read_file tool.
        assert "content already loaded in conversation" in body
        assert "read_file" not in body
        assert any(
            block.get("type") == "image" for block in _blocks(json.loads(body))
        )

    @pytest.mark.parametrize(
        ("model", "formatter"),
        [
            ("system.ai.claude-sonnet-5", "anthropic"),
            ("system.ai.gpt-6-sol", "openai"),
            ("system.ai.gemini-3-5-flash", "gemini"),
        ],
    )
    def test_tasks_and_crews_find_the_supported_file_types(
        self, model: str, formatter: str
    ) -> None:
        from crewai_files import get_supported_content_types

        llm = LLM(model=f"databricks/{model}")

        assert llm._multimodal_formatter_name() == formatter
        assert "image/" in get_supported_content_types(
            llm._multimodal_formatter_name(), llm.api
        )


class TestCheckpointRestore:
    @pytest.mark.parametrize(
        ("model", "cls"),
        [
            ("system.ai.claude-sonnet-5", DatabricksAnthropicCompletion),
            ("system.ai.gpt-6-sol", DatabricksCompletion),
            ("system.ai.gemini-3-5-flash", DatabricksGeminiCompletion),
        ],
    )
    def test_serialized_llms_restore_as_databricks(
        self, monkeypatch: pytest.MonkeyPatch, gateway: Gateway, model: str, cls: type
    ) -> None:
        from crewai.agents.agent_builder.base_agent import _validate_llm_ref

        monkeypatch.setenv("DATABRICKS_HOST", HOST)
        monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-env")
        dumped = LLM(model=f"databricks/{model}", api_key="dapi-secret").model_dump()

        assert "dapi-secret" not in json.dumps(dumped, default=str)

        restored = _validate_llm_ref(dumped)

        assert type(restored) is cls
        assert restored.model == model
        restored.call("hi")
        assert gateway.requests[0].url.host == httpx.URL(HOST).host
        assert gateway.requests[0].headers["Authorization"] == "Bearer dapi-env"

    def test_custom_service_keeps_its_api(self) -> None:
        from crewai.agents.agent_builder.base_agent import _validate_llm_ref

        dumped = LLM(model="databricks/main.ai.service", api="gemini").model_dump()

        assert type(_validate_llm_ref(dumped)) is DatabricksGeminiCompletion


class TestNameBasedDefaults:
    def test_claude_output_limit_comes_from_the_model(self) -> None:
        from crewai.llms.providers.anthropic.completion import (
            _default_max_tokens_for_model,
        )

        llm = LLM(model="databricks/system.ai.claude-sonnet-5")

        assert llm.max_tokens == _default_max_tokens_for_model("claude-sonnet-5")
        assert "max_tokens" not in llm.to_config_dict()

    def test_an_explicit_generic_default_output_limit_is_kept(self) -> None:
        llm = LLM(model="databricks/system.ai.claude-sonnet-5", max_tokens=32000)

        assert llm.to_config_dict()["max_tokens"] == 32000
        assert LLM(**llm.to_config_dict()).max_tokens == 32000

    def test_explicit_output_limit_wins(self) -> None:
        llm = LLM(model="databricks/system.ai.claude-sonnet-5", max_tokens=100)

        assert llm.max_tokens == 100
        assert llm.to_config_dict()["max_tokens"] == 100

    @pytest.mark.parametrize(
        ("model", "thinking"),
        [
            ("system.ai.gemini-2-5-flash", True),
            ("system.ai.gemini-3-5-flash", True),
            ("system.ai.gemini-2-0-flash", False),
        ],
    )
    def test_gemini_thinking_follows_the_model_version(
        self, model: str, thinking: bool
    ) -> None:
        llm = LLM(model=f"databricks/{model}")

        assert (llm.thinking_config is not None) is thinking

    def test_learned_settings_follow_a_client_params_base_url(self) -> None:
        llm = LLM(
            model="databricks/system.ai.gpt-6-sol",
            databricks_auth=_auth(),
            client_params={"base_url": "https://proxy.example.com/v1"},
        )

        assert llm._reasoning_effort_key()[0] == "https://proxy.example.com/v1"

    def test_learned_model_settings_are_keyed_by_workspace(self) -> None:
        llm = LLM(model="databricks/system.ai.gpt-6-sol", databricks_auth=_auth())

        assert llm._reasoning_effort_key() == (
            f"{HOST}/ai-gateway/mlflow/v1",
            "system.ai.gpt-6-sol",
        )


class TestInterceptors:
    def test_gemini_accepts_an_interceptor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from crewai.llms.hooks.base import BaseInterceptor
        from crewai.llms.hooks.transport import AsyncHTTPTransport, HTTPTransport

        class Passthrough(BaseInterceptor[httpx.Request, httpx.Response]):
            def on_outbound(self, message: httpx.Request) -> httpx.Request:
                return message

            def on_inbound(self, message: httpx.Response) -> httpx.Response:
                return message

        transports: list[Any] = []

        def record(auth: DatabricksAuth, *, transport: Any = None, **_: Any) -> Any:
            transports.append(transport)
            return httpx.Client()

        def record_async(auth: DatabricksAuth, *, transport: Any = None, **_: Any) -> Any:
            transports.append(transport)
            return httpx.AsyncClient()

        monkeypatch.setattr(gemini_completion, "build_httpx_client", record)
        monkeypatch.setattr(gemini_completion, "build_async_httpx_client", record_async)
        interceptor = Passthrough()

        llm = LLM(
            model="databricks/system.ai.gemini-3-5-flash",
            interceptor=interceptor,
            databricks_auth=_auth(),
        )
        llm._get_sync_client()

        assert llm.interceptor is interceptor
        assert isinstance(transports[0], HTTPTransport)
        assert isinstance(transports[1], AsyncHTTPTransport)


def test_gemini_rejects_an_interceptor_of_the_wrong_type() -> None:
    with pytest.raises(ImportError, match="BaseInterceptor"):
        LLM(model="databricks/system.ai.gemini-3-5-flash", interceptor="not-an-interceptor")


class TestSharedCredentials:
    def test_copies_made_before_the_first_call_share_credentials(self) -> None:
        import copy

        llm = LLM(model="databricks/system.ai.gpt-6-sol", api_key="dapi-x", host=HOST)

        assert isinstance(llm.databricks_auth, DatabricksAuth)
        assert llm.databricks_token is None  # moved into the shared DatabricksAuth
        assert copy.deepcopy(llm).databricks_auth is llm.databricks_auth
        assert llm.model_copy().databricks_auth is llm.databricks_auth


class TestRealTransportChain:
    @pytest.mark.parametrize(
        "model",
        ["system.ai.claude-sonnet-5", "system.ai.gpt-6-sol", "system.ai.gemini-3-5-flash"],
    )
    def test_default_clients_sign_and_send_through_the_gateway_transport(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """The real build_httpx_client chain, with only the socket layer mocked."""
        recorder = Gateway()

        def handle(self: Any, request: httpx.Request) -> httpx.Response:
            return recorder(request)

        monkeypatch.setattr(gateway_module._EnvProxyTransport, "handle_request", handle)
        llm = LLM(model=f"databricks/{model}", databricks_auth=_auth())

        assert llm.call("hi") == "OK"

        request = recorder.requests[0]
        assert request.url.path.startswith("/ai-gateway/")
        assert request.headers["Authorization"] == "Bearer tok-1"
        assert request.headers["User-Agent"].startswith(f"{USER_AGENT_PRODUCT}/")

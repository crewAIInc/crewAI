import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

pytest.importorskip("anthropic")

from anthropic import Anthropic

from crewai.llm import LLM

ENDPOINT = "https://res.services.ai.azure.com"
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "Hello from Foundry"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


STREAM = "".join(
    [
        _sse(
            "message_start",
            {"type": "message_start", "message": {**MESSAGE, "content": []}},
        ),
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello from Foundry"},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AZURE_API_KEY",
        "AZURE_ENDPOINT",
        "AZURE_API_BASE",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def requests_seen() -> list[httpx.Request]:
    return []


@pytest.fixture
def make_llm(requests_seen: list[httpx.Request]) -> Callable[..., LLM]:
    """Build the LLM, then give its sync client a mock transport.

    The client is the real Anthropic SDK built from the class's own client
    parameters, so the URL, key and headers under test are the ones it derived.
    """

    def build(response: httpx.Response | None = None, **kwargs: Any) -> LLM:
        def handler(request: httpx.Request) -> httpx.Response:
            requests_seen.append(request)
            return response or httpx.Response(200, json=MESSAGE)

        llm = LLM(
            model="azure/claude-sonnet-4-6",
            endpoint=ENDPOINT,
            api_key="foundry-key",
            **kwargs,
        )
        params = llm._get_client_params()
        params["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
        llm._client = Anthropic(**params)
        return llm

    return build


def test_call_posts_to_the_foundry_messages_endpoint(
    make_llm: Callable[..., LLM], requests_seen: list[httpx.Request]
) -> None:
    result = make_llm().call("Say hello")

    assert result == "Hello from Foundry"
    (request,) = requests_seen
    assert request.method == "POST"
    assert str(request.url) == f"{ENDPOINT}/anthropic/v1/messages"
    assert "chat/completions" not in request.url.path


def test_request_uses_foundry_key_header_and_api_version(
    make_llm: Callable[..., LLM], requests_seen: list[httpx.Request]
) -> None:
    make_llm().call("Say hello")

    (request,) = requests_seen
    assert request.headers["x-api-key"] == "foundry-key"
    assert request.headers["anthropic-version"] == "2023-06-01"


def test_deployment_name_is_sent_as_model(
    make_llm: Callable[..., LLM], requests_seen: list[httpx.Request]
) -> None:
    make_llm().call("Say hello")

    body = json.loads(requests_seen[0].content)
    assert body["model"] == "claude-sonnet-4-6"


def test_system_message_is_sent_in_the_system_field(
    make_llm: Callable[..., LLM], requests_seen: list[httpx.Request]
) -> None:
    make_llm().call(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Say hello"},
        ]
    )

    body = json.loads(requests_seen[0].content)
    assert body["system"] == "Be brief."
    assert [m["role"] for m in body["messages"]] == ["user"]


def test_tools_are_sent_in_anthropic_format(
    make_llm: Callable[..., LLM], requests_seen: list[httpx.Request]
) -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }

    make_llm().call("Weather in Paris?", tools=[tool])

    (sent,) = json.loads(requests_seen[0].content)["tools"]
    assert sent["name"] == "get_weather"
    assert sent["input_schema"]["properties"] == {"city": {"type": "string"}}


def test_streaming_returns_the_assembled_text(
    make_llm: Callable[..., LLM], requests_seen: list[httpx.Request]
) -> None:
    response = httpx.Response(
        200, content=STREAM, headers={"content-type": "text/event-stream"}
    )

    result = make_llm(response, stream=True).call("Say hello")

    assert result == "Hello from Foundry"
    assert json.loads(requests_seen[0].content)["stream"] is True
    assert str(requests_seen[0].url) == f"{ENDPOINT}/anthropic/v1/messages"


def test_tool_use_response_runs_the_matching_function(
    make_llm: Callable[..., LLM],
) -> None:
    tool_use = {
        **MESSAGE,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        ],
        "stop_reason": "tool_use",
    }
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
    called: list[str] = []

    def get_weather(city: str) -> str:
        called.append(city)
        return "sunny"

    result = make_llm(httpx.Response(200, json=tool_use)).call(
        "Weather in Paris?",
        tools=[tool],
        available_functions={"get_weather": get_weather},
    )

    assert called == ["Paris"]
    assert result == "sunny"

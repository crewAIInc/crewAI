"""Gateways that report upstream failures inside an HTTP 200 body.

OpenRouter (and other OpenAI-compatible gateways) commit ``200 OK`` as soon as the
upstream provider accepts the request, so a later provider failure arrives as an
``error`` object with no ``choices``. Without a guard the absent ``choices`` reaches
the OpenAI SDK's parse helper and surfaces as
``TypeError: 'NoneType' object is not iterable``, naming neither the provider, the
status, nor the fact that a timeout happened.

These tests drive the real OpenAI SDK over ``httpx.MockTransport``, so the parse
path under test is the one that runs in production. No network is involved.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel

from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.openai_compatible.completion import (
    OpenAICompatibleCompletion,
)


BASE_URL = "https://openrouter.ai/api/v1"


class Answer(BaseModel):
    """Minimal structured-output target."""

    a: str


def _error_envelope(message: str, code: object) -> dict[str, Any]:
    """A 200 body holding only an upstream error, as gateways send it."""
    return {"error": {"message": message, "code": code}, "id": "gen-abc123"}


def _completion_body(content: str) -> dict[str, Any]:
    return {
        "id": "gen-ok",
        "object": "chat.completion",
        "created": 1,
        "model": "z-ai/glm-5.3",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content, "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


def _tool_call_body() -> dict[str, Any]:
    body = _completion_body("")
    body["choices"][0]["finish_reason"] = "tool_calls"
    body["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "refusal": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
            }
        ],
    }
    return body


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (payload + "data: [DONE]\n\n").encode()


def _stream_error_chunk() -> dict[str, Any]:
    """OpenRouter's documented mid-stream error shape: error plus an empty delta."""
    return {
        "id": "gen-abc123",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "z-ai/glm-5.3",
        "provider": "Z.AI",
        "error": {"code": 504, "message": "The operation was aborted"},
        "choices": [
            {"index": 0, "delta": {"content": ""}, "finish_reason": "error"}
        ],
    }


def _make_llm(
    responder: Any,
    *,
    cls: type[OpenAICompletion] = OpenAICompletion,
    stream: bool = False,
    **kwargs: Any,
) -> OpenAICompletion:
    """Build a provider whose SDK clients are pinned to a mock transport.

    Replaces the private clients rather than patching module globals so the test
    leaves no state behind for whatever runs next.
    """
    llm = cls(model=kwargs.pop("model", "z-ai/glm-5.3"), api_key="sk-test", stream=stream, **kwargs)
    transport = httpx.MockTransport(responder)
    llm._client = openai.OpenAI(
        api_key="sk-test", base_url=BASE_URL, http_client=httpx.Client(transport=transport)
    )
    llm._async_client = openai.AsyncOpenAI(
        api_key="sk-test",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(transport=transport),
    )
    return llm


def _json_responder(body: dict[str, Any], status: int = 200) -> Any:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return respond


# --------------------------------------------------------------------------- #
# The reported defect, across the four non-streaming paths
# --------------------------------------------------------------------------- #


def test_sync_structured_surfaces_upstream_timeout_not_typeerror() -> None:
    """The reported case: response_model= through a gateway that timed out."""
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))

    with pytest.raises(
        openai.InternalServerError,
        match=r"z-ai/glm-5\.3 via openrouter\.ai returned HTTP 200 with an upstream "
        r"error and no choices: The operation was aborted \(upstream code 504\)",
    ):
        llm.call("hi", response_model=Answer)


def test_sync_plain_surfaces_upstream_timeout_not_typeerror() -> None:
    """Not only structured calls: the plain path had the same defect."""
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))

    with pytest.raises(
        openai.InternalServerError,
        match=r"returned HTTP 200 with an upstream error and no choices: "
        r"The operation was aborted \(upstream code 504\)",
    ):
        llm.call("hi")


@pytest.mark.asyncio
async def test_async_structured_surfaces_upstream_timeout() -> None:
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))

    with pytest.raises(
        openai.InternalServerError,
        match=r"The operation was aborted \(upstream code 504\)",
    ):
        await llm.acall("hi", response_model=Answer)


@pytest.mark.asyncio
async def test_async_plain_surfaces_upstream_timeout() -> None:
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))

    with pytest.raises(
        openai.InternalServerError,
        match=r"The operation was aborted \(upstream code 504\)",
    ):
        await llm.acall("hi")


def test_error_message_never_mentions_nonetype() -> None:
    """The whole point: the raised error must be actionable."""
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))

    with pytest.raises(openai.APIStatusError) as caught:
        llm.call("hi", response_model=Answer)

    text = str(caught.value)
    assert "NoneType" not in text
    assert "z-ai/glm-5.3" in text
    assert "openrouter.ai" in text
    assert "504" in text


# --------------------------------------------------------------------------- #
# Upstream status codes map to the class the SDK uses for the honest error
# --------------------------------------------------------------------------- #


def test_upstream_429_raises_rate_limit_error() -> None:
    """A masked rate limit stays catchable as openai.RateLimitError."""
    llm = _make_llm(_json_responder(_error_envelope("Slow down", 429)))

    with pytest.raises(openai.RateLimitError, match="Slow down"):
        llm.call("hi")


def test_upstream_502_raises_internal_server_error() -> None:
    llm = _make_llm(_json_responder(_error_envelope("Bad gateway", 502)))

    with pytest.raises(openai.InternalServerError, match=r"Bad gateway"):
        llm.call("hi")


def test_upstream_400_raises_bad_request_error() -> None:
    llm = _make_llm(_json_responder(_error_envelope("Malformed schema", 400)))

    with pytest.raises(openai.BadRequestError, match="Malformed schema"):
        llm.call("hi")


def test_upstream_error_body_is_attached_to_exception() -> None:
    """Callers inspecting `.body` get the gateway's own error object."""
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))

    with pytest.raises(openai.InternalServerError) as caught:
        llm.call("hi")

    assert caught.value.body == {"message": "The operation was aborted", "code": 504}
    assert caught.value.status_code == 200


def test_string_status_code_is_understood() -> None:
    """Some gateways stringify the upstream status."""
    llm = _make_llm(_json_responder(_error_envelope("Slow down", "429")))

    with pytest.raises(openai.RateLimitError, match=r"upstream code 429"):
        llm.call("hi")


def test_openai_style_slug_code_is_not_treated_as_a_status() -> None:
    """`code` is a slug in OpenAI-style bodies, not an HTTP status."""
    llm = _make_llm(_json_responder(_error_envelope("no such model", "model_not_found")))

    with pytest.raises(openai.APIResponseValidationError, match="no such model"):
        llm.call("hi")

    # and the message must not invent a status code
    llm2 = _make_llm(_json_responder(_error_envelope("no such model", "model_not_found")))
    with pytest.raises(openai.APIResponseValidationError) as caught:
        llm2.call("hi")
    assert "upstream code" not in str(caught.value)


def test_error_without_message_still_names_the_model() -> None:
    llm = _make_llm(_json_responder({"error": {"code": 504}, "id": "gen-x"}))

    with pytest.raises(openai.InternalServerError, match="no message given"):
        llm.call("hi")


# --------------------------------------------------------------------------- #
# Malformed 200s that carry no error object either
# --------------------------------------------------------------------------- #


def test_missing_choices_without_error_object_is_a_validation_error() -> None:
    """Distinguishable from a retryable upstream failure, which is the report's ask."""
    llm = _make_llm(_json_responder({"id": "gen-x", "object": "chat.completion"}))

    with pytest.raises(
        openai.APIResponseValidationError,
        match="returned HTTP 200 with no choices and no error object",
    ):
        llm.call("hi")


def test_empty_choices_list_is_reported_not_indexerror() -> None:
    """`choices: []` used to reach `choices[0]` and raise IndexError."""
    body = _completion_body("hi")
    body["choices"] = []
    llm = _make_llm(_json_responder(body))

    with pytest.raises(
        openai.APIResponseValidationError, match="no choices and no error object"
    ):
        llm.call("hi")


# --------------------------------------------------------------------------- #
# Existing behaviour that must not change
# --------------------------------------------------------------------------- #


def test_happy_path_plain_completion_unchanged() -> None:
    llm = _make_llm(_json_responder(_completion_body("hello there")))

    assert llm.call("hi") == "hello there"


def test_happy_path_structured_completion_unchanged() -> None:
    llm = _make_llm(_json_responder(_completion_body('{"a": "hi"}')))

    result = llm.call("hi", response_model=Answer)

    assert isinstance(result, Answer)
    assert result.a == "hi"


@pytest.mark.asyncio
async def test_happy_path_async_plain_unchanged() -> None:
    llm = _make_llm(_json_responder(_completion_body("hello there")))

    assert await llm.acall("hi") == "hello there"


@pytest.mark.asyncio
async def test_happy_path_async_structured_unchanged() -> None:
    llm = _make_llm(_json_responder(_completion_body('{"a": "hi"}')))

    result = await llm.acall("hi", response_model=Answer)

    assert isinstance(result, Answer)
    assert result.a == "hi"


def test_token_usage_still_tracked_on_happy_path() -> None:
    """with_raw_response must not cost us usage accounting."""
    llm = _make_llm(_json_responder(_completion_body("hello there")))

    llm.call("hi")

    usage = llm.get_token_usage_summary()
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 4, 7)


def test_tool_calls_still_returned_to_the_caller() -> None:
    """The tool-call branch reads `.choices` too and must be unaffected."""
    llm = _make_llm(_json_responder(_tool_call_body()))

    result = llm.call("what is the weather in Lisbon?")

    assert isinstance(result, list)
    assert result[0].function.name == "get_weather"


def test_tool_execution_follow_up_turn_unchanged() -> None:
    """Tool result -> follow-up completion, the path after a tool executes."""
    calls: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=_tool_call_body())
        return httpx.Response(200, json=_completion_body("It is sunny in Lisbon"))

    llm = _make_llm(respond)

    result = llm.call(
        "what is the weather in Lisbon?",
        available_functions={"get_weather": lambda city: f"sunny in {city}"},
    )

    assert result == "sunny in Lisbon"


def test_real_http_error_status_still_raises_its_own_type() -> None:
    """A genuine 429 was already handled; it must not route through the new guard."""
    llm = _make_llm(
        _json_responder({"error": {"message": "rate limited"}}, status=429)
    )

    with pytest.raises(openai.RateLimitError):
        llm.call("hi")


def test_length_finish_reason_still_raises_through_raw_response() -> None:
    """`with_raw_response` must not drop the SDK's own parse-time errors."""
    body = _completion_body('{"a": "hi"}')
    body["choices"][0]["finish_reason"] = "length"
    llm = _make_llm(_json_responder(body))

    with pytest.raises(Exception) as caught:
        llm.call("hi", response_model=Answer)

    assert isinstance(caught.value, openai.LengthFinishReasonError)


def test_non_json_200_body_is_left_to_existing_handling() -> None:
    """Our guard must not turn a non-JSON 200 into a JSONDecodeError."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"<html>gateway boom</html>", headers={"content-type": "text/html"}
        )

    llm = _make_llm(respond)

    with pytest.raises(Exception) as caught:
        llm.call("hi")

    assert not isinstance(caught.value, json.JSONDecodeError)


def test_masked_404_does_not_trigger_the_responses_api_fallback() -> None:
    """`_is_responses_only_error` keys off OpenAI's wording, not any 404.

    A gateway 404 inside a 200 must not send the call to /v1/responses.
    """
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json=_error_envelope("No endpoints found", 404))

    llm = _make_llm(respond)

    with pytest.raises(ValueError, match="No endpoints found"):
        llm.call("hi")

    assert all("/responses" not in path for path in seen), seen


# --------------------------------------------------------------------------- #
# Streaming already had a guard in the SDK; pin that it stays intact
# --------------------------------------------------------------------------- #


def test_sync_streaming_error_chunk_already_surfaces_upstream_message() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse([_stream_error_chunk()]),
            headers={"content-type": "text/event-stream"},
        )

    llm = _make_llm(respond, stream=True)

    with pytest.raises(Exception, match="The operation was aborted"):
        llm.call("hi")


@pytest.mark.asyncio
async def test_async_streaming_error_chunk_already_surfaces_upstream_message() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse([_stream_error_chunk()]),
            headers={"content-type": "text/event-stream"},
        )

    llm = _make_llm(respond, stream=True)

    with pytest.raises(Exception, match="The operation was aborted"):
        await llm.acall("hi")


# --------------------------------------------------------------------------- #
# The subclass users actually reach OpenRouter through
# --------------------------------------------------------------------------- #


def test_openai_compatible_subclass_inherits_the_guard() -> None:
    """OpenRouter routes through OpenAICompatibleCompletion, not OpenAICompletion."""
    llm = _make_llm(
        _json_responder(_error_envelope("The operation was aborted", 504)),
        cls=OpenAICompatibleCompletion,
        model="z-ai/glm-5.3",
        provider="openrouter",
    )

    with pytest.raises(
        openai.InternalServerError,
        match=r"The operation was aborted \(upstream code 504\)",
    ):
        llm.call("hi", response_model=Answer)


# --------------------------------------------------------------------------- #
# Every entry in the status table, and what the message is allowed to contain
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("upstream_code", "expected"),
    [
        (400, openai.BadRequestError),
        (401, openai.AuthenticationError),
        (403, openai.PermissionDeniedError),
        (404, openai.NotFoundError),
        (409, openai.ConflictError),
        (422, openai.UnprocessableEntityError),
        (429, openai.RateLimitError),
        (500, openai.InternalServerError),
        (503, openai.InternalServerError),
        (504, openai.InternalServerError),
    ],
)
def test_every_upstream_code_maps_to_its_sdk_exception(
    upstream_code: int, expected: type[Exception]
) -> None:
    """A masked failure must be catchable exactly like the honest one."""
    llm = _make_llm(_json_responder(_error_envelope("upstream said no", upstream_code)))

    # 404 is rewritten to ValueError by the provider's own model-not-found handler,
    # which is the same thing it does for a real 404.
    if expected is openai.NotFoundError:
        with pytest.raises(ValueError, match="upstream said no"):
            llm.call("hi")
        return

    with pytest.raises(expected, match=f"upstream code {upstream_code}"):
        llm.call("hi")


def test_message_names_the_host_but_never_url_credentials() -> None:
    """The message reaches logs and the model context, so it carries only the host.

    Some gateways accept credentials in the URL; the guard reports
    `request.url.host` rather than the URL so those are not echoed.
    """
    llm = _make_llm(_json_responder(_error_envelope("The operation was aborted", 504)))
    llm._client = openai.OpenAI(
        api_key="sk-test",
        base_url="https://user:sk-secret123@gw.example/api/v1?api_key=sk-leak456",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                _json_responder(_error_envelope("The operation was aborted", 504))
            )
        ),
    )

    with pytest.raises(openai.InternalServerError) as caught:
        llm.call("hi")

    text = str(caught.value)
    assert "gw.example" in text
    assert "sk-secret123" not in text
    assert "sk-leak456" not in text

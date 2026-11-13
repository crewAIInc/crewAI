"""GPT-5.6 rejects function tools alongside `reasoning_effort` on completions.

The family applies a server-side `reasoning_effort` default, so a payload that
never mentions the parameter is still refused as soon as tools are attached --
measured against the live endpoint:

    gpt-5.6-sol  tools, no reasoning_effort key   -> 400
    gpt-5.6-sol  tools, reasoning_effort="none"   -> OK
    gpt-5.5      tools, no reasoning_effort key   -> OK

So an ordinary agent with a tool fails on this model family and nothing else. The
400 is recovered by resending with an explicit "none"; no model list is involved,
so a family OpenAI restricts later is handled without a release here.

GPT-6 Astra and GPT-6.1 Sol offer "none" in the same 400 and then refuse it, so
their tools are served only by /v1/responses (live endpoint, 2026-11-03):

    gpt-6.1-sol  tools, no reasoning_effort key   -> 400, offers "none"
    gpt-6.1-sol  tools, reasoning_effort="none"   -> 400, unsupported value
    gpt-6.1-sol  tools on /v1/responses           -> OK
    gpt-6-luna   tools, reasoning_effort="none"   -> OK (gpt-6.1-terra too)

Those calls move to the Responses API, and the model is remembered so its later
calls go there directly.
"""

import gzip
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from crewai import Agent, Crew, Task
from crewai.llms.providers.openai.completion import (
    _LEARNED_RESPONSES_ONLY_MODELS,
    _LEARNED_TOOLS_WITHOUT_REASONING_MODELS,
    OpenAICompletion,
)
from crewai.tools import tool


MESSAGES = [{"role": "user", "content": "hi"}]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


def build(model: str = "gpt-5.6-sol", **kwargs) -> OpenAICompletion:
    return OpenAICompletion(model=model, api_key="sk-test", **kwargs)


def _bad_request(message: str, param: str = "reasoning_effort") -> BadRequestError:
    body = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
        }
    }
    return BadRequestError(
        message,
        response=httpx.Response(
            400,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body=body,
    )


def tools_effort_error(model: str = "gpt-5.6-sol") -> BadRequestError:
    """The 400 OpenAI returns for tools + reasoning_effort."""
    return _bad_request(
        f"Function tools with reasoning_effort are not supported for {model} in "
        "/v1/chat/completions. To use function tools, use /v1/responses or set "
        "reasoning_effort to 'none'."
    )


def unsupported_value_error() -> BadRequestError:
    """What o1/o3 return for reasoning_effort="none" -- not recoverable this way."""
    return _bad_request("Unsupported value: 'reasoning_effort' does not support 'none'.")


def none_rejected_error() -> BadRequestError:
    """GPT-6 Astra and GPT-6.1 Sol refusing the "none" retry (live wording)."""
    body = {
        "error": {
            "message": "Unsupported value: 'reasoning_effort' does not support "
            "'none' with this model. Supported values are: 'low', 'medium', "
            "'high', and 'xhigh'.",
            "type": "invalid_request_error",
            "param": "reasoning_effort",
            "code": "unsupported_value",
        }
    }
    return BadRequestError(
        body["error"]["message"],
        response=httpx.Response(
            400,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        ),
        body=body,
    )


def responses_only_tools_error() -> BadRequestError:
    """GPT-6 Astra on Bedrock (live wording): no "none" way out, and it rejects it."""
    return _bad_request(
        "Function tools with reasoning_effort are not supported for "
        "us.openai.gpt-6-astra in /v1/chat/completions. To use function tools, use "
        "/v1/responses."
    )


@pytest.fixture(autouse=True)
def _clear_learned_models():
    """Keep the process-wide learned sets from leaking between tests."""
    _LEARNED_RESPONSES_ONLY_MODELS.clear()
    _LEARNED_TOOLS_WITHOUT_REASONING_MODELS.clear()
    yield
    _LEARNED_RESPONSES_ONLY_MODELS.clear()
    _LEARNED_TOOLS_WITHOUT_REASONING_MODELS.clear()


class TestErrorDetection:
    def test_matches_the_tools_effort_400(self):
        assert OpenAICompletion._rejects_reasoning_effort_with_tools(
            tools_effort_error()
        )

    def test_matches_a_flat_error_body(self):
        """The SDK populates `body` flat or nested depending on how it parsed."""
        message = (
            "Function tools with reasoning_effort are not supported for gpt-5.6-sol."
        )
        body = {"message": message, "param": "reasoning_effort"}
        error = BadRequestError(
            message,
            response=httpx.Response(
                400, json=body, request=httpx.Request("POST", "https://x/v1/c")
            ),
            body=body,
        )

        assert OpenAICompletion._rejects_reasoning_effort_with_tools(error)

    def test_ignores_the_unsupported_value_400(self):
        """o1/o3 reject reasoning_effort="none", so retrying with it can't help."""
        assert not OpenAICompletion._rejects_reasoning_effort_with_tools(
            unsupported_value_error()
        )

    def test_ignores_unrelated_exceptions(self):
        assert not OpenAICompletion._rejects_reasoning_effort_with_tools(
            RuntimeError("boom")
        )

    @pytest.mark.parametrize(
        ("error", "offers_none"),
        [
            (tools_effort_error("gpt-6.1-sol"), True),
            (responses_only_tools_error(), False),
        ],
        ids=["openai", "bedrock"],
    )
    def test_spots_whether_the_tools_400_offers_none(self, error, offers_none):
        assert OpenAICompletion._offers_reasoning_effort_none(error) is offers_none

    @pytest.mark.parametrize(
        ("error", "rejects_none"),
        [
            (none_rejected_error(), True),
            (unsupported_value_error(), True),
            (tools_effort_error("gpt-6.1-sol"), False),
            (
                _bad_request("Unsupported value: 'temperature'.", param="temperature"),
                False,
            ),
            (RuntimeError("Unsupported value: 'reasoning_effort'"), False),
        ],
        ids=["gpt-6", "flat-body", "tools-400", "other-param", "not-a-400"],
    )
    def test_spots_the_none_retry_being_refused(self, error, rejects_none):
        assert OpenAICompletion._rejects_reasoning_effort_none(error) is rejects_none


class TestRetryParams:
    def test_sets_an_explicit_none(self):
        """Absence means "use the server default", which is what was rejected."""
        params = build()._reasoning_effort_none_params({"model": "gpt-5.6-sol"})

        assert params is not None
        assert params["reasoning_effort"] == "none"

    def test_overrides_a_requested_effort(self):
        params = build()._reasoning_effort_none_params({"reasoning_effort": "high"})

        assert params is not None
        assert params["reasoning_effort"] == "none"

    def test_returns_none_when_already_none(self):
        """Nothing left to try -- the retry must not loop."""
        assert (
            build()._reasoning_effort_none_params({"reasoning_effort": "none"}) is None
        )


class TestRetryBehaviour:
    def test_retries_with_none_and_succeeds(self, monkeypatch):
        llm = build()
        seen: list[dict] = []

        def fake_handle(params, **kwargs):
            seen.append(params)
            if params.get("reasoning_effort") != "none":
                raise tools_effort_error()
            return "ok"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "ok"
        assert len(seen) == 2, "expected one rejected call and one retry"
        assert "reasoning_effort" not in seen[0]
        assert seen[1]["reasoning_effort"] == "none"
        assert seen[1]["tools"], "tools must survive the retry"

    def test_does_not_retry_forever(self, monkeypatch):
        """A model that fails even with "none" must surface, not loop."""
        llm = build()
        calls: list[dict] = []

        def always_fail(params, **kwargs):
            calls.append(params)
            raise tools_effort_error()

        monkeypatch.setattr(llm, "_handle_completion", always_fail)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 2, "one original call plus exactly one retry"

    def test_does_not_retry_an_unrelated_400(self, monkeypatch):
        llm = build("o3-mini")
        calls: list[dict] = []

        def fail(params, **kwargs):
            calls.append(params)
            raise unsupported_value_error()

        monkeypatch.setattr(llm, "_handle_completion", fail)

        with pytest.raises(BadRequestError):
            llm._call_completions(MESSAGES, tools=TOOLS)

        assert len(calls) == 1

    @pytest.mark.parametrize(
        "rejection",
        [tools_effort_error(), none_rejected_error()],
        ids=["tools-400", "none-refused"],
    )
    def test_recovered_call_reports_no_failure(self, monkeypatch, rejection):
        """The retried attempt must not emit LLMCallFailedEvent."""
        llm = build()
        emitted: list[str] = []
        monkeypatch.setattr(
            llm,
            "_emit_call_failed_event",
            lambda **kwargs: emitted.append(kwargs.get("error", "")),
        )

        class FakeCompletions:
            def create(self, **kwargs):
                raise rejection

            @property
            def with_raw_response(self):
                """The provider reads the raw body to spot 200-wrapped errors."""
                return self

        monkeypatch.setattr(
            llm,
            "_get_sync_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=FakeCompletions())
            ),
        )

        # The original BadRequestError must survive for _call_completions to match.
        with pytest.raises(BadRequestError):
            llm._handle_completion({"model": "gpt-5.6-sol", "messages": MESSAGES})

        assert emitted == []

    @pytest.mark.asyncio
    async def test_async_path_retries_too(self, monkeypatch):
        llm = build()
        seen: list[dict] = []

        async def fake_handle(params, **kwargs):
            seen.append(params)
            if params.get("reasoning_effort") != "none":
                raise tools_effort_error()
            return "ok"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_handle)

        assert await llm._acall_completions(MESSAGES, tools=TOOLS) == "ok"
        assert seen[1]["reasoning_effort"] == "none"


def _accepts_tools_only_without_reasoning(params: dict, **kwargs):
    """Chat completions as GPT-6 Luna serves it: tools need reasoning_effort="none"."""
    if params.get("tools") and params.get("reasoning_effort") != "none":
        raise tools_effort_error("gpt-6-luna")
    return "ok"


class TestToolsWithoutReasoningAreRemembered:
    """The rejected round trip is paid once per model and endpoint, not per call."""

    def test_the_next_tool_call_sends_none_up_front(self, monkeypatch):
        seen: list[dict] = []

        def handle(params, **kwargs):
            seen.append(params)
            return _accepts_tools_only_without_reasoning(params)

        for _ in range(2):
            llm = build("gpt-6-luna")
            monkeypatch.setattr(llm, "_handle_completion", handle)
            assert llm._call_completions(MESSAGES, tools=TOOLS) == "ok"

        assert [p.get("reasoning_effort") for p in seen] == [None, "none", "none"]

    def test_a_streamed_tool_call_learns_it_too(self, monkeypatch):
        seen: list[dict] = []

        def handle(params, **kwargs):
            seen.append(params)
            return _accepts_tools_only_without_reasoning(params)

        for _ in range(2):
            llm = build("gpt-6-luna", stream=True)
            monkeypatch.setattr(llm, "_handle_streaming_completion", handle)
            assert llm._call_completions(MESSAGES, tools=TOOLS) == "ok"

        assert [p.get("reasoning_effort") for p in seen] == [None, "none", "none"]

    @pytest.mark.asyncio
    async def test_the_async_path_learns_it_too(self, monkeypatch):
        seen: list[dict] = []

        async def handle(params, **kwargs):
            seen.append(params)
            return _accepts_tools_only_without_reasoning(params)

        for _ in range(2):
            llm = build("gpt-6-luna")
            monkeypatch.setattr(llm, "_ahandle_completion", handle)
            assert await llm._acall_completions(MESSAGES, tools=TOOLS) == "ok"

        assert [p.get("reasoning_effort") for p in seen] == [None, "none", "none"]

    def test_a_call_without_tools_keeps_its_reasoning(self, monkeypatch):
        llm = build("gpt-6-luna")
        monkeypatch.setattr(llm, "_handle_completion", _accepts_tools_only_without_reasoning)
        llm._call_completions(MESSAGES, tools=TOOLS)

        params = build("gpt-6-luna")._prepare_completion_params(MESSAGES)

        assert "reasoning_effort" not in params

    def test_a_lesson_from_one_endpoint_stays_there(self, monkeypatch):
        llm = build("gpt-6-luna", base_url="https://gateway.example/v1")
        monkeypatch.setattr(llm, "_handle_completion", _accepts_tools_only_without_reasoning)
        llm._call_completions(MESSAGES, tools=TOOLS)

        params = build("gpt-6-luna")._prepare_completion_params(MESSAGES, tools=TOOLS)

        assert "reasoning_effort" not in params

    def test_a_model_that_refuses_none_is_not_recorded(self, monkeypatch):
        llm = build("gpt-6.1-sol")
        monkeypatch.setattr(llm, "_handle_completion", _rejects_tools_like_gpt_6_1_sol)
        monkeypatch.setattr(llm, "_call_responses", lambda **kwargs: "via responses")

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "via responses"
        assert not _LEARNED_TOOLS_WITHOUT_REASONING_MODELS


def _rejects_tools_like_gpt_6_1_sol(params: dict, **kwargs):
    """Chat completions as GPT-6.1 Sol serves it: no tool call ever succeeds."""
    if params.get("reasoning_effort") == "none":
        raise none_rejected_error()
    raise tools_effort_error("gpt-6.1-sol")


TOOL_ROUND_TRIP = [
    {"role": "user", "content": "What is 17 times 23?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "multiply", "arguments": '{"a": 17, "b": 23}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "391"},
]


class TestToolsOnlyOnResponses:
    """GPT-6 Astra and GPT-6.1 Sol: tool calls move to the Responses API."""

    def test_sync_call_moves_to_responses(self, monkeypatch):
        llm = build("gpt-6.1-sol")
        chat: list[dict] = []
        responses: list[dict] = []

        def fake_chat(params, **kwargs):
            chat.append(params)
            _rejects_tools_like_gpt_6_1_sol(params)

        def fake_responses(params, **kwargs):
            responses.append(params)
            return "391"

        monkeypatch.setattr(llm, "_handle_completion", fake_chat)
        monkeypatch.setattr(llm, "_handle_responses", fake_responses)

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "391"
        assert [p.get("reasoning_effort") for p in chat] == [None, "none"]
        assert [t["name"] for t in responses[0]["tools"]] == ["multiply"]
        assert responses[0]["store"] is False, "chat completions stores nothing"
        assert llm._effective_api() == "responses"

    @pytest.mark.asyncio
    async def test_async_call_moves_to_responses(self, monkeypatch):
        llm = build("gpt-6-astra")
        chat: list[dict] = []
        responses: list[dict] = []

        async def fake_chat(params, **kwargs):
            chat.append(params)
            _rejects_tools_like_gpt_6_1_sol(params)

        async def fake_responses(params, **kwargs):
            responses.append(params)
            return "391"

        monkeypatch.setattr(llm, "_ahandle_completion", fake_chat)
        monkeypatch.setattr(llm, "_ahandle_responses", fake_responses)

        assert await llm._acall_completions(MESSAGES, tools=TOOLS) == "391"
        assert [p.get("reasoning_effort") for p in chat] == [None, "none"]
        assert [t["name"] for t in responses[0]["tools"]] == ["multiply"]
        assert llm._effective_api() == "responses"

    def test_sync_stream_moves_to_responses_stream(self, monkeypatch):
        llm = build("gpt-6.1-sol", stream=True)
        responses: list[dict] = []

        def fake_responses(params, **kwargs):
            responses.append(params)
            return "391"

        monkeypatch.setattr(
            llm, "_handle_streaming_completion", _rejects_tools_like_gpt_6_1_sol
        )
        monkeypatch.setattr(llm, "_handle_streaming_responses", fake_responses)

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "391"
        assert responses[0]["stream"] is True
        assert responses[0]["tools"]

    @pytest.mark.asyncio
    async def test_async_stream_moves_to_responses_stream(self, monkeypatch):
        llm = build("gpt-6.1-sol", stream=True)
        responses: list[dict] = []

        async def fake_chat(params, **kwargs):
            _rejects_tools_like_gpt_6_1_sol(params)

        async def fake_responses(params, **kwargs):
            responses.append(params)
            return "391"

        monkeypatch.setattr(llm, "_ahandle_streaming_completion", fake_chat)
        monkeypatch.setattr(llm, "_ahandle_streaming_responses", fake_responses)

        assert await llm._acall_completions(MESSAGES, tools=TOOLS) == "391"
        assert responses[0]["stream"] is True
        assert responses[0]["tools"]

    def test_tool_result_turn_goes_straight_to_responses(self, monkeypatch):
        """The turn after a tool runs pays no rejected chat call and keeps the result."""
        llm = build("gpt-6.1-sol")
        chat: list[dict] = []
        responses: list[dict] = []

        def fake_chat(params, **kwargs):
            chat.append(params)
            _rejects_tools_like_gpt_6_1_sol(params)

        def fake_responses(params, **kwargs):
            responses.append(params)
            return "391"

        monkeypatch.setattr(llm, "_handle_completion", fake_chat)
        monkeypatch.setattr(llm, "_handle_responses", fake_responses)

        llm.call(MESSAGES, tools=TOOLS)
        chat.clear()
        assert llm.call(TOOL_ROUND_TRIP, tools=TOOLS) == "391"

        assert chat == []
        items = responses[-1]["input"]
        assert {"type": "function_call", "call_id": "call_1"}.items() <= items[1].items()
        assert items[2] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "391",
        }

    @pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-6.1-terra"])
    def test_models_that_accept_none_stay_on_chat(self, monkeypatch, model):
        llm = build(model)
        seen: list[dict] = []

        def fake_chat(params, **kwargs):
            seen.append(params)
            if params.get("reasoning_effort") != "none":
                raise tools_effort_error(model)
            return "391"

        def no_responses(params, **kwargs):
            raise AssertionError("must not reach the Responses API")

        monkeypatch.setattr(llm, "_handle_completion", fake_chat)
        monkeypatch.setattr(llm, "_handle_responses", no_responses)

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "391"
        assert seen[-1]["reasoning_effort"] == "none"
        assert llm._effective_api() == "completions"

    def test_a_retry_failing_for_another_reason_surfaces(self, monkeypatch):
        """Only the refused "none" moves the call; anything else is the caller's."""
        llm = build("gpt-6.1-sol")
        calls: list[dict] = []

        def fake_chat(params, **kwargs):
            calls.append(params)
            if params.get("reasoning_effort") == "none":
                raise _bad_request("Invalid 'messages'.", param="messages")
            raise tools_effort_error("gpt-6.1-sol")

        def no_responses(params, **kwargs):
            raise AssertionError("must not reach the Responses API")

        monkeypatch.setattr(llm, "_handle_completion", fake_chat)
        monkeypatch.setattr(llm, "_handle_responses", no_responses)

        with pytest.raises(BadRequestError, match="Invalid 'messages'"):
            llm._call_completions(MESSAGES, tools=TOOLS)
        assert len(calls) == 2
        assert llm._effective_api() == "completions"

    @pytest.mark.parametrize("custom_openai", [False, True])
    def test_skips_the_none_retry_when_it_is_not_offered(
        self, monkeypatch, custom_openai
    ):
        """Bedrock names only /v1/responses, so that is where the call goes."""
        llm = build(
            "us.openai.gpt-6-astra",
            custom_openai=custom_openai,
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
        )
        seen: list[dict] = []

        def fake_chat(params, **kwargs):
            seen.append(params)
            raise responses_only_tools_error()

        monkeypatch.setattr(llm, "_handle_completion", fake_chat)
        monkeypatch.setattr(llm, "_handle_responses", lambda params, **kw: "391")

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "391"
        assert len(seen) == 1, "no reasoning_effort='none' retry"

    def test_a_compatible_server_is_not_remembered(self, monkeypatch):
        """Its later calls still start on chat completions, as `api` asks."""
        llm = build(
            "us.openai.gpt-6-astra",
            custom_openai=True,
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
        )
        monkeypatch.setattr(llm, "_handle_completion", _rejects_tools_like_gpt_6_1_sol)
        monkeypatch.setattr(llm, "_handle_responses", lambda params, **kw: "391")

        assert llm._call_completions(MESSAGES, tools=TOOLS) == "391"
        assert llm._effective_api() == "completions"


class TestLearnedPerEndpoint:
    """What one endpoint taught never reroutes the same model name elsewhere."""

    GATEWAY = "https://gateway.example/v1"

    def test_a_gateway_lesson_stays_on_that_gateway(self):
        build("gpt-6.1-sol", base_url=self.GATEWAY)._remember_responses_only_model()

        assert build("gpt-6.1-sol", base_url=self.GATEWAY)._effective_api() == (
            "responses"
        )
        assert build("gpt-6.1-sol")._effective_api() == "completions"
        assert (
            build("gpt-6.1-sol", base_url="https://other.example/v1")._effective_api()
            == "completions"
        )

    def test_an_openai_lesson_does_not_reroute_a_gateway(self):
        build("gpt-6.1-sol")._remember_responses_only_model()

        assert build("gpt-6.1-sol", base_url=self.GATEWAY)._effective_api() == (
            "completions"
        )


class TestStoreOnTheMoveToResponses:
    """Chat completions stores nothing by default, and the move must not change that."""

    def test_an_implicit_move_asks_openai_not_to_store(self):
        params = build("gpt-6.1-sol")._prepare_responses_params(MESSAGES)

        assert params["store"] is False

    @pytest.mark.parametrize("store", [True, False])
    def test_an_explicit_store_wins(self, store):
        params = build("gpt-6.1-sol", store=store)._prepare_responses_params(MESSAGES)

        assert params["store"] is store

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"api": "responses"},
            {"auto_chain": True},
            {"previous_response_id": "resp_1"},
            {"base_url": "https://gateway.example/v1"},
        ],
        ids=["asked-for-responses", "auto-chain", "chained", "compatible-server"],
    )
    def test_leaves_store_to_the_server_otherwise(self, kwargs):
        """Chaining needs a stored response, and a gateway may not know the field."""
        params = build("gpt-6.1-sol", **kwargs)._prepare_responses_params(MESSAGES)

        assert "store" not in params


def _streamed_function_call() -> list[SimpleNamespace]:
    """The Responses API events for one streamed `multiply` call."""
    item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="multiply",
        arguments='{"a": 17, "b": 23}',
    )
    response = SimpleNamespace(
        id="resp_1", status="completed", output=[item], usage=None
    )
    return [
        SimpleNamespace(type="response.created", response=response),
        SimpleNamespace(type="response.output_item.done", item=item),
        SimpleNamespace(type="response.completed", response=response),
    ]


class TestStreamedToolCalls:
    """A native agent passes no `available_functions` and runs the calls itself."""

    def test_sync_stream_returns_the_tool_calls(self, monkeypatch):
        llm = build("gpt-6.1-sol", api="responses", stream=True)
        events = _streamed_function_call()
        monkeypatch.setattr(
            llm,
            "_get_sync_client",
            lambda: SimpleNamespace(
                responses=SimpleNamespace(create=lambda **kwargs: iter(events))
            ),
        )

        result = llm._handle_streaming_responses({"input": MESSAGES})

        assert result == [
            {"id": "call_1", "name": "multiply", "arguments": '{"a": 17, "b": 23}'}
        ]

    @pytest.mark.asyncio
    async def test_async_stream_returns_the_tool_calls(self, monkeypatch):
        llm = build("gpt-6.1-sol", api="responses", stream=True)
        events = _streamed_function_call()

        async def stream():
            for event in events:
                yield event

        async def create(**kwargs):
            return stream()

        monkeypatch.setattr(
            llm,
            "_get_async_client",
            lambda: SimpleNamespace(responses=SimpleNamespace(create=create)),
        )

        result = await llm._ahandle_streaming_responses({"input": MESSAGES})

        assert result == [
            {"id": "call_1", "name": "multiply", "arguments": '{"a": 17, "b": 23}'}
        ]

    def test_given_the_functions_it_still_runs_them(self, monkeypatch):
        llm = build("gpt-6.1-sol", api="responses", stream=True)
        events = _streamed_function_call()
        monkeypatch.setattr(
            llm,
            "_get_sync_client",
            lambda: SimpleNamespace(
                responses=SimpleNamespace(create=lambda **kwargs: iter(events))
            ),
        )

        result = llm._handle_streaming_responses(
            {"input": MESSAGES}, available_functions={"multiply": lambda a, b: a * b}
        )

        assert result == "391"


@pytest.fixture(scope="module")
def vcr_config(vcr_config):
    """Record 400s too: the cassette below exists to show the recovery from two.

    conftest skips recording any response >= 400. These carry only an error body,
    so they are kept with every header but the content type dropped.
    """
    filter_response = vcr_config["before_record_response"]

    def keep_rejections(response):
        if response["status"]["code"] != 400:
            return filter_response(response)
        body = response["body"]["string"]
        if isinstance(body, bytes) and body.startswith(b"\x1f\x8b"):
            body = gzip.decompress(body).decode()
        return {
            **response,
            "headers": {"content-type": ["application/json"]},
            "body": {"string": body},
        }

    return {**vcr_config, "before_record_response": keep_rejections}


@pytest.mark.vcr()
def test_gpt_6_1_sol_agent_runs_its_tool(caplog):
    """Recorded live: two refused chat calls, then the tool round trip on /v1/responses."""

    looked_up: list[str] = []

    @tool("lookup_code")
    def lookup_code(name: str) -> str:
        """Look up the access code for a project by name."""
        looked_up.append(name)
        return f"{name.upper()}-ZETA-7741"

    agent = Agent(
        role="Records clerk",
        goal="Report access codes exactly as the lookup tool returns them",
        backstory="You never guess a code.",
        llm=OpenAICompletion(model="gpt-6.1-sol"),
        tools=[lookup_code],
    )
    task = Task(
        description="What is the access code for the project named 'orion'?",
        expected_output="The access code, exactly as the tool returned it",
        agent=agent,
    )

    with caplog.at_level(logging.ERROR):
        result = Crew(agents=[agent], tasks=[task]).kickoff()

    # Cassettes match on path alone, so a reply could be played to the wrong
    # call; the tool having run is what proves the round trip.
    assert looked_up == ["orion"]
    assert "ORION-ZETA-7741" in str(result)
    assert caplog.records == [], "a recovered call must not log an error"


@pytest.mark.vcr()
def test_gpt_6_1_sol_streaming_agent_runs_its_tool(caplog):
    """Recorded live with stream=True: the streamed tool call reaches the agent."""

    looked_up: list[str] = []

    @tool("lookup_code")
    def lookup_code(name: str) -> str:
        """Look up the access code for a project by name."""
        looked_up.append(name)
        return f"{name.upper()}-ZETA-7741"

    agent = Agent(
        role="Records clerk",
        goal="Report access codes exactly as the lookup tool returns them",
        backstory="You never guess a code.",
        llm=OpenAICompletion(model="gpt-6.1-sol", stream=True),
        tools=[lookup_code],
    )
    task = Task(
        description="What is the access code for the project named 'vega'?",
        expected_output="The access code, exactly as the tool returned it",
        agent=agent,
    )

    with caplog.at_level(logging.ERROR):
        result = Crew(agents=[agent], tasks=[task]).kickoff()

    # Cassettes match on path alone, so a reply could be played to the wrong
    # call; the tool having run is what proves the round trip.
    assert looked_up == ["vega"]
    assert "VEGA-ZETA-7741" in str(result)
    assert caplog.records == [], "a recovered call must not log an error"


class TestAgentDefinitions:
    """The bar: ordinary agent definitions run, with tools and with reasoning."""

    @staticmethod
    def _stub_transport(llm: OpenAICompletion, monkeypatch) -> list[dict]:
        """Reject like GPT-5.6 until an explicit reasoning_effort="none" arrives.

        The reasoning planner calls the LLM with its own `available_functions`, so
        the stub answers that request with the planner's JSON contract and every
        other request with the final answer.
        """
        sent: list[dict] = []

        def fake_handle(params, available_functions=None, **kwargs):
            sent.append(params)
            if params.get("tools") and params.get("reasoning_effort") != "none":
                raise tools_effort_error()
            if available_functions and "create_reasoning_plan" in available_functions:
                return json.dumps(
                    {"plan": "Multiply 17 by 23.", "steps": [], "ready": True}
                )
            return "391"

        monkeypatch.setattr(llm, "_handle_completion", fake_handle)
        return sent

    @staticmethod
    def _multiply_tool():
        @tool("multiply")
        def multiply(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        return multiply

    def test_agent_with_tools(self, monkeypatch):
        llm = build()
        sent = self._stub_transport(llm, monkeypatch)

        agent = Agent(
            role="Math",
            goal="Answer correctly",
            backstory="You are precise.",
            llm=llm,
            tools=[self._multiply_tool()],
        )
        task = Task(
            description="What is 17 times 23?",
            expected_output="A number",
            agent=agent,
        )

        assert "391" in str(Crew(agents=[agent], tasks=[task]).kickoff())
        assert any(p.get("reasoning_effort") == "none" for p in sent)

    def test_agent_with_tools_and_reasoning(self, monkeypatch):
        """`reasoning=True` adds a planning call, which must recover as well."""
        llm = build()
        sent = self._stub_transport(llm, monkeypatch)

        agent = Agent(
            role="Math",
            goal="Answer correctly",
            backstory="You are precise.",
            llm=llm,
            tools=[self._multiply_tool()],
            reasoning=True,
        )
        task = Task(
            description="What is 17 times 23?",
            expected_output="A number",
            agent=agent,
        )

        assert "391" in str(Crew(agents=[agent], tasks=[task]).kickoff())
        assert any(p.get("reasoning_effort") == "none" for p in sent)

    def test_agent_without_tools_is_untouched(self, monkeypatch):
        """No tools means no conflict, so nothing should be rewritten."""
        llm = build()
        sent = self._stub_transport(llm, monkeypatch)

        agent = Agent(
            role="Math",
            goal="Answer correctly",
            backstory="You are precise.",
            llm=llm,
        )
        task = Task(
            description="What is 17 times 23?",
            expected_output="A number",
            agent=agent,
        )

        assert "391" in str(Crew(agents=[agent], tasks=[task]).kickoff())
        assert all("reasoning_effort" not in p for p in sent)

"""Empty Chat Completions must retain the provider's termination metadata.

Drive the real SDK over a mock HTTP transport; no API keys or network are needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import openai
import pytest
from pydantic import BaseModel

from crewai.llms.base_llm import LLMEmptyResponseError
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.utilities.agent_utils import aget_llm_response, get_llm_response
from crewai.utilities.i18n import I18N_DEFAULT
from crewai_core.printer import Printer


USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 3,
    "total_tokens": 13,
    "completion_tokens_details": {"reasoning_tokens": 0},
}
MESSAGES = [
    {"role": "user", "content": "Say hello using reply."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_reply",
                "type": "function",
                "function": {"name": "reply", "arguments": '{"text":"hello"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_reply", "content": '{"status":"success"}'},
]


def make_llm(
    *,
    stream: bool,
    finish_reason: str | None = "stop",
    content: str | None = "",
    with_usage: bool = True,
    tool_calls: list[dict[str, Any]] | None = None,
    stream_deltas: list[dict[str, Any]] | None = None,
) -> tuple[OpenAICompletion, list[httpx.Request]]:
    """Build a real OpenAI SDK client backed by deterministic HTTP responses."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Return completion JSON or SSE chunks without network access."""
        requests.append(request)
        common = {"id": "chatcmpl-empty", "created": 1, "model": "gpt-4o-mini"}
        if not stream:
            return httpx.Response(
                200,
                json={
                    **common,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "tool_calls": tool_calls,
                            },
                        }
                    ],
                    "usage": USAGE if with_usage else None,
                },
            )
        delta: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            delta["tool_calls"] = [
                {"index": i, **call} for i, call in enumerate(tool_calls)
            ]
        chunks = [
            {
                **common,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
            }
        ]
        if stream_deltas is not None:
            chunks = [
                {
                    **common,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": part,
                            "finish_reason": finish_reason
                            if index == len(stream_deltas) - 1
                            else None,
                        }
                    ],
                }
                for index, part in enumerate(stream_deltas)
            ]
        if with_usage:
            chunks.append(
                {
                    **common,
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": USAGE,
                }
            )
        data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(
            200,
            content=(data + "data: [DONE]\n\n").encode(),
            headers={"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(respond)
    llm = OpenAICompletion(model="gpt-4o-mini", api_key="sk-test", stream=stream)
    llm._client = openai.OpenAI(
        api_key="sk-test", max_retries=0, http_client=httpx.Client(transport=transport)
    )
    llm._async_client = openai.AsyncOpenAI(
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=transport),
    )
    return llm, requests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length", "content_filter", None])
@pytest.mark.asyncio
async def test_empty_response_exposes_provider_metadata(
    stream: bool, is_async: bool, finish_reason: str | None
) -> None:
    """Preserve termination metadata across sync, async, and streaming calls."""
    llm, requests = make_llm(stream=stream, finish_reason=finish_reason)
    with pytest.raises(LLMEmptyResponseError) as caught:
        if is_async:
            await llm.acall(MESSAGES)
        else:
            llm.call(MESSAGES)
    assert caught.value.finish_reason == finish_reason
    assert caught.value.response_id == "chatcmpl-empty"
    assert caught.value.usage["reasoning_tokens"] == 0
    assert caught.value.usage["completion_tokens"] == 3
    assert len(requests) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("with_post_tool_reasoning", [False, True])
@pytest.mark.asyncio
async def test_tool_loop_accepts_stop_after_tool_result(
    stream: bool, is_async: bool, with_post_tool_reasoning: bool
) -> None:
    """Accept an empty stop after a tool, including the synthetic follow-up."""
    llm, requests = make_llm(stream=stream)
    messages = list(MESSAGES)
    if with_post_tool_reasoning:
        messages.append(
            {"role": "user", "content": I18N_DEFAULT.slice("post_tool_reasoning")}
        )
    kwargs = {
        "llm": llm,
        "messages": messages,
        "callbacks": [],
        "printer": Printer(),
        "executor_context": SimpleNamespace(
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
            messages=messages,
        ),
        "verbose": False,
    }
    result = (
        await aget_llm_response(**kwargs) if is_async else get_llm_response(**kwargs)
    )
    assert result == ""
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("finish_reason", "last_role"),
    [("length", "tool"), ("stop", "user")],
)
def test_empty_turn_is_only_accepted_after_a_stop_tool_result(
    finish_reason: str, last_role: str
) -> None:
    """Reject truncation and empty responses without a qualifying tool result."""
    llm, requests = make_llm(stream=False, finish_reason=finish_reason)
    messages = list(MESSAGES)
    if last_role == "user":
        messages[-1] = {"role": "user", "content": "continue"}
    context = SimpleNamespace(
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
        messages=messages,
    )

    with pytest.raises(LLMEmptyResponseError):
        get_llm_response(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            executor_context=context,
            verbose=False,
        )
    assert len(requests) == 1


def test_native_executor_finishes_after_tool_result_empty_turn() -> None:
    """Finish the native executor after a successful tool and empty stop."""
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.tools.base_tool import BaseTool, to_langchain

    class ReplyTool(BaseTool):
        name: str = "reply"
        description: str = "Send a reply"

        def _run(self, text: str) -> str:
            """Simulate delivery of a reply and return a successful tool result."""
            return '{"status":"success"}'

    llm = Mock()
    llm.call.side_effect = [
        [
            {
                "id": "call_reply",
                "type": "function",
                "function": {"name": "reply", "arguments": '{"text":"hello"}'},
            }
        ],
        LLMEmptyResponseError(
            finish_reason="stop",
            response_id="chatcmpl-empty",
            usage={"total_tokens": 3},
        ),
    ]
    tool = ReplyTool()
    executor = CrewAgentExecutor(
        tools=to_langchain([tool]),
        original_tools=[tool],
    )
    executor.llm = llm
    executor.agent = SimpleNamespace(
        id="agent-id",
        key="test-agent",
        role="tester",
        verbose=False,
        fingerprint=None,
        tools_results=[],
    )
    executor.task = SimpleNamespace(name="test-task", description="test", id="task-id")
    executor.messages = [{"role": "user", "content": "Say hello"}]
    executor.callbacks = []
    executor.iterations = 0
    executor.max_iter = 3
    executor.request_within_rpm_limit = None
    executor.respect_context_window = False

    result = executor._invoke_loop_native_tools()

    assert type(result).__name__ == "AgentFinish"
    assert result.output == ""
    assert result.text == ""
    assert llm.call.call_count == 2
    assert [message["role"] for message in executor.messages] == [
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
    ]


def test_experimental_native_executor_finishes_after_tool_result_empty_turn() -> None:
    """Finish the experimental executor after a tool and empty stop."""
    from crewai.experimental.agent_executor import AgentExecutor
    from crewai.tools.base_tool import BaseTool, to_langchain

    class ReplyTool(BaseTool):
        name: str = "reply"
        description: str = "Send a reply"

        def _run(self, text: str) -> str:
            """Simulate delivery of a reply and return a successful tool result."""
            return '{"status":"success"}'

    llm = Mock()
    llm.call.side_effect = [
        [
            {
                "id": "call_reply",
                "type": "function",
                "function": {"name": "reply", "arguments": '{"text":"hello"}'},
            }
        ],
        LLMEmptyResponseError(
            finish_reason="stop",
            response_id="chatcmpl-empty",
            usage={"total_tokens": 3},
        ),
    ]
    tool = ReplyTool()
    executor = AgentExecutor()
    executor.agent = SimpleNamespace(
        id="agent-id",
        key="test-agent",
        role="tester",
        verbose=False,
        planning_enabled=False,
        tools_results=[],
    )
    executor.task = SimpleNamespace(name="test-task", description="test", id="task-id")
    executor.llm = llm
    executor.original_tools = [tool]
    executor.tools = to_langchain([tool])
    executor._setup_native_tools()
    executor.messages = [{"role": "user", "content": "Say hello"}]
    executor.state.use_native_tools = True
    executor.callbacks = []
    executor.request_within_rpm_limit = None

    assert executor.call_llm_native_tools() == "native_tool_calls"
    assert executor.execute_native_tool() == "native_tool_completed"
    assert executor.call_llm_native_tools() == "native_finished"
    assert type(executor.state.current_answer).__name__ == "AgentFinish"
    assert executor.state.current_answer.output == ""
    assert executor.state.current_answer.text == ""
    assert llm.call.call_count == 2
    assert [message["role"] for message in executor.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_first_turn_empty_response_remains_an_error(
    stream: bool, is_async: bool
) -> None:
    """Keep first-call empty responses distinguishable from successful output."""
    llm, requests = make_llm(stream=stream, content=None, with_usage=False)
    with pytest.raises(LLMEmptyResponseError) as caught:
        if is_async:
            await llm.acall("hello")
        else:
            llm.call("hello")
    assert type(caught.value).__name__ == "LLMEmptyResponseError"
    assert "reasoning_tokens" not in caught.value.usage
    assert len(requests) == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("tool_call", [False, True])
@pytest.mark.asyncio
async def test_nonempty_text_and_tool_calls_are_unchanged(
    stream: bool, is_async: bool, tool_call: bool
) -> None:
    """Preserve normal text and tool-call responses."""
    calls = MESSAGES[1]["tool_calls"] if tool_call else None
    llm, requests = make_llm(
        stream=stream,
        content=None if tool_call else "hello",
        tool_calls=calls,
        finish_reason="tool_calls" if tool_call else "stop",
    )
    result = await llm.acall("hello") if is_async else llm.call("hello")
    assert (
        isinstance(result, list) and len(result) == 1
        if tool_call
        else result == "hello"
    )
    assert len(requests) == 1


class StructuredReply(BaseModel):
    """A structured response used to exercise the SDK parsing paths."""

    text: str


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("with_usage", [False, True])
@pytest.mark.parametrize("content", [None, ""])
@pytest.mark.asyncio
async def test_structured_stream_empty_response(
    is_async: bool, with_usage: bool, content: str | None
) -> None:
    """Empty structured streams must expose metadata on the first call."""
    llm, requests = make_llm(stream=True, with_usage=with_usage, content=content)
    with pytest.raises(LLMEmptyResponseError) as caught:
        if is_async:
            await llm.acall("hello", response_model=StructuredReply)
        else:
            llm.call("hello", response_model=StructuredReply)
    assert caught.value.finish_reason == "stop"
    assert caught.value.response_id == "chatcmpl-empty"
    if with_usage:
        assert caught.value.usage["completion_tokens"] == 3
    else:
        assert "reasoning_tokens" not in caught.value.usage
    assert len(requests) == 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_structured_stream_valid_response(is_async: bool) -> None:
    """Valid structured streams continue returning the requested model."""
    llm, requests = make_llm(stream=True, content='{"text":"hello"}')
    result = (
        await llm.acall("hello", response_model=StructuredReply)
        if is_async
        else llm.call("hello", response_model=StructuredReply)
    )
    assert result == StructuredReply(text="hello")
    assert len(requests) == 1


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", None])
@pytest.mark.asyncio
async def test_async_structured_empty_termination_metadata(
    finish_reason: str | None,
) -> None:
    """Structured async failures retain non-stop and absent finish reasons."""
    llm, requests = make_llm(stream=True, finish_reason=finish_reason)
    with pytest.raises(LLMEmptyResponseError) as caught:
        await llm.acall("hello", response_model=StructuredReply)
    assert caught.value.finish_reason == finish_reason
    assert caught.value.response_id == "chatcmpl-empty"
    assert caught.value.usage["completion_tokens"] == 3
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("finish_reason", "error_type"),
    [
        ("length", openai.LengthFinishReasonError),
        ("content_filter", openai.ContentFilterFinishReasonError),
    ],
)
def test_sync_structured_sdk_failures_remain_errors(
    finish_reason: str, error_type: type[Exception]
) -> None:
    """Preserve the SDK's explicit truncation and content-filter exceptions."""
    llm, requests = make_llm(stream=True, finish_reason=finish_reason)
    with pytest.raises(error_type):
        llm.call("hello", response_model=StructuredReply)
    assert len(requests) == 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_structured_stream_tool_turn_finishes(is_async: bool) -> None:
    """The structured path accepts an empty terminal turn after tool execution."""
    llm, requests = make_llm(stream=True)
    messages = list(MESSAGES)
    kwargs = {
        "llm": llm,
        "messages": messages,
        "callbacks": [],
        "printer": Printer(),
        "executor_context": SimpleNamespace(
            before_llm_call_hooks=[], after_llm_call_hooks=[], messages=messages
        ),
        "response_model": StructuredReply,
        "verbose": False,
    }
    result = (
        await aget_llm_response(**kwargs) if is_async else get_llm_response(**kwargs)
    )
    assert result == ""
    assert len(requests) == 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("with_text", [False, True])
@pytest.mark.asyncio
async def test_structured_stream_preserves_tool_calls(
    is_async: bool, execute: bool, with_text: bool
) -> None:
    """Fragmented tool calls take precedence over structured text and run once."""
    llm, requests = make_llm(
        stream=True,
        finish_reason="tool_calls",
        stream_deltas=[
            {
                "content": '{"text":"planning"}' if with_text else None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_reply",
                        "type": "function",
                        "function": {"name": "reply", "arguments": '{"text":'},
                    }
                ],
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '"hello"}'}}]},
        ],
    )
    reply = Mock(return_value="delivered")
    kwargs = {
        "response_model": StructuredReply,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "reply",
                    "description": "Send a reply",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "available_functions": {"reply": reply} if execute else None,
    }
    result = (
        await llm.acall("hello", **kwargs) if is_async else llm.call("hello", **kwargs)
    )
    if execute:
        assert result == "delivered"
        reply.assert_called_once_with(text="hello")
    else:
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "call_reply"
        assert result[0]["function"] == {
            "name": "reply",
            "arguments": '{"text":"hello"}',
        }
        reply.assert_not_called()
    assert len(requests) == 1
    assert llm.get_token_usage_summary().total_tokens == USAGE["total_tokens"]

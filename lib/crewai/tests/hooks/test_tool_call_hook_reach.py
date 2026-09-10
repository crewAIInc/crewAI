"""Which tool calls the ``pre_tool_call`` hooks can see.

The executor paths (ReAct, native function calling, LiteAgent) dispatch
``PRE_TOOL_CALL`` before a tool body runs, so a hook that returns ``False``
keeps the body from running there. Three paths reached a tool body without
that dispatch: a declarative Flow ``do: call: tool`` action, and the tool
wrappers the OpenAI-agents and LangGraph adapters hand to their frameworks.
A policy registered once with ``@on(InterceptionPoint.PRE_TOOL_CALL)`` was
silently skipped on all three.

Each path is exercised directly, without a model, and must behave like the
executor paths: the hook sees the call exactly once with the tool input, a
deny keeps the body from running and reaches the caller as the blocked
message, and a ``post_tool_call`` rewrite reaches the caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from crewai.flow import Flow
from crewai.hooks.dispatch import InterceptionPoint, clear_all, on
from crewai.tools import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name
import pytest


BODY_CALLS: list[str] = []


class EchoTool(BaseTool):
    name: str = "echo_tool"
    description: str = "Echoes its input."

    def _run(self, text: str) -> str:
        BODY_CALLS.append(text)
        return f"echo:{text}"


TOOL_NAME = sanitize_tool_name("echo_tool")


def _invoke_flow_tool_action(text: str) -> Any:
    yaml_str = f"""
schema: crewai.flow/v1
name: EchoFlow
methods:
  echo:
    do:
      call: tool
      ref: {__name__}:EchoTool
      with:
        text: {text}
    start: true
"""
    return Flow.from_declaration(contents=yaml_str).kickoff()


def _invoke_openai_agents_wrapper(text: str) -> Any:
    pytest.importorskip("agents")
    from crewai.agents.agent_adapters.openai_agents.openai_agent_tool_adapter import (
        OpenAIAgentToolAdapter,
    )

    adapter = OpenAIAgentToolAdapter()
    adapter.configure_tools([EchoTool()])
    (function_tool,) = adapter.converted_tools
    return asyncio.run(function_tool.on_invoke_tool(None, {"text": text}))


def _invoke_langgraph_wrapper(text: str) -> Any:
    pytest.importorskip("langchain_core")
    from crewai.agents.agent_adapters.langgraph.langgraph_tool_adapter import (
        LangGraphToolAdapter,
    )

    adapter = LangGraphToolAdapter()
    adapter.configure_tools([EchoTool()])
    (structured_tool,) = adapter.converted_tools
    return asyncio.run(structured_tool.func(text=text))


PATHS: list[tuple[str, Callable[[str], Any]]] = [
    ("flow tool action", _invoke_flow_tool_action),
    ("openai agents adapter", _invoke_openai_agents_wrapper),
    ("langgraph adapter", _invoke_langgraph_wrapper),
]


@pytest.fixture(autouse=True)
def _clean() -> Any:
    clear_all()
    BODY_CALLS.clear()
    yield
    clear_all()
    BODY_CALLS.clear()


@pytest.fixture
def seen() -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    @on(InterceptionPoint.PRE_TOOL_CALL)
    def record(ctx: Any) -> None:
        calls.append((ctx.tool_name, dict(ctx.tool_input)))

    return calls


@pytest.mark.parametrize(("_label", "invoke"), PATHS, ids=[p[0] for p in PATHS])
def test_the_call_is_seen_exactly_once_with_its_input(_label, invoke, seen):
    assert invoke("hi") == "echo:hi"

    assert seen == [(TOOL_NAME, {"text": "hi"})]
    assert BODY_CALLS == ["hi"]


@pytest.mark.parametrize(("_label", "invoke"), PATHS, ids=[p[0] for p in PATHS])
def test_a_deny_keeps_the_body_from_running(_label, invoke):
    @on(InterceptionPoint.PRE_TOOL_CALL)
    def deny(ctx: Any) -> bool:
        return False

    result = invoke("hi")

    assert BODY_CALLS == []
    assert result == f"Tool execution blocked by hook. Tool: {TOOL_NAME}"


@pytest.mark.parametrize(("_label", "invoke"), PATHS, ids=[p[0] for p in PATHS])
def test_a_post_hook_rewrite_reaches_the_caller(_label, invoke):
    @on(InterceptionPoint.POST_TOOL_CALL)
    def redact(ctx: Any) -> str:
        return "[redacted]"

    assert invoke("hi") == "[redacted]"
    assert BODY_CALLS == ["hi"]

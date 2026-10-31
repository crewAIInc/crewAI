"""ReAct text-path tool hooks must retain aborts and use effective cache inputs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from crewai.agents.agent_executor import AgentExecutor
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, clear_all, on
from crewai.tools import BaseTool
from crewai.tools.base_tool import to_langchain


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."

    def _run(self, term: str = "default") -> str:
        return f"value:{term}"


@pytest.fixture(autouse=True)
def clean_hooks():
    clear_all()
    yield
    crewai_event_bus.flush()
    clear_all()


def _react_text_executor() -> tuple[AgentExecutor, AgentAction]:
    """Wire an executor for the ReAct text tool route (execute_tool_action)."""
    tool = LookupTool()
    structured = to_langchain([tool])
    handler = ToolsHandler(cache=CacheHandler())
    executor = AgentExecutor(
        tools=structured, original_tools=[tool], tools_handler=handler
    )
    action = AgentAction(
        thought="I should look this up",
        tool="lookup",
        tool_input='{"term": "hello"}',
        text=(
            "Thought: I should look this up\n"
            'Action: lookup\nAction Input: {"term": "hello"}'
        ),
    )
    executor.state.current_answer = action
    return executor, action


def test_post_tool_abort_reaches_caller_on_react_text_path():
    abort = HookAborted("policy denied", source="test-policy")

    @on(InterceptionPoint.POST_TOOL_CALL)
    def deny(_context):
        raise abort

    executor, action = _react_text_executor()

    with pytest.raises(HookAborted) as exc:
        executor.execute_tool_action()

    assert exc.value is abort
    # The abort must not be folded into a recoverable observation for the LLM.
    assert "Error executing tool" not in action.text


def test_plain_error_still_becomes_recovery_observation():
    """Only deliberate aborts re-raise; plain errors stay recoverable."""
    executor, action = _react_text_executor()

    with patch(
        "crewai.agents.agent_executor.execute_tool_and_check_finality",
        side_effect=RuntimeError("boom"),
    ):
        result = executor.execute_tool_action()

    assert result == "tool_completed"
    assert "Error executing tool: boom" in action.text

"""ReAct text-path tool hooks must retain aborts and use effective cache inputs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from crewai.agents.agent_executor import AgentExecutor
from typing import Literal

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, clear_all, on
from crewai.utilities.tool_utils import aexecute_tool_and_check_finality
from crewai.tools import BaseTool
from crewai.tools.base_tool import to_langchain


class _MinimalAgent(BaseAgent):
    """A concrete BaseAgent: enough for the tool-execution event path."""

    entity_type: Literal["agent"] = "agent"

    def execute_task(self, task):  # pragma: no cover - not used on this path
        return ""

    async def aexecute_task(self, task):  # pragma: no cover
        return ""

    def create_agent_executor(self, *args, **kwargs):  # pragma: no cover
        return None

    def get_delegation_tools(self, crew_agents):  # pragma: no cover
        return []

    def get_mcp_tools(self):  # pragma: no cover
        return []

    def get_platform_tools(self):  # pragma: no cover
        return []


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
    agent = _MinimalAgent(role="tester", goal="test goal", backstory="test backstory")
    executor = AgentExecutor(
        tools=structured, original_tools=[tool], tools_handler=handler, agent=agent
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


def test_before_hook_edits_apply_to_empty_action_input_on_react_text_path():
    """#8852: a before_tool_call hook edit must reach the tool even when the
    model called it with an empty Action Input."""
    executor, _ = _react_text_executor()
    executor.state.current_answer = AgentAction(
        thought="lookup",
        tool="lookup",
        tool_input="{}",
        text="Thought: lookup\nAction: lookup\nAction Input: {}",
    )

    @on(InterceptionPoint.PRE_TOOL_CALL)
    def fill_default(context):
        context.tool_input["term"] = "effective"

    with patch(
        "crewai.events.event_bus.CrewAIEventsBus.register_entity",
        lambda self, entity: None,
    ):
        executor.execute_tool_action()

    assert 'Observation: value:effective' in executor.state.current_answer.text


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_before_hook_edits_apply_to_empty_action_input_on_async_path():
    """#8852: the async path (aexecute_tool_and_check_finality) behaves the
    same for empty Action Input."""
    tool = LookupTool()
    structured = to_langchain([tool])
    action = AgentAction(
        thought="lookup",
        tool="lookup",
        tool_input="{}",
        text="Thought: lookup\nAction: lookup\nAction Input: {}",
    )

    @on(InterceptionPoint.PRE_TOOL_CALL)
    def fill_default(context):
        context.tool_input["term"] = "effective"

    result = await aexecute_tool_and_check_finality(
        action, structured, tools_handler=ToolsHandler(cache=CacheHandler())
    )

    assert "value:effective" in result.result

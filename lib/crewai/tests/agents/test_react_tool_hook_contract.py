"""ReAct text-path tool hooks must retain aborts and use effective cache inputs."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from crewai import Agent
from crewai.agents.agent_executor import AgentExecutor
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import AgentAction
from crewai.agents.tools_handler import ToolsHandler
from crewai.events import crewai_event_bus
from crewai.hooks.dispatch import HookAborted, InterceptionPoint, clear_all, on
from crewai.llms.base_llm import BaseLLM
from crewai.tools import BaseTool
from crewai.tools.base_tool import to_langchain
from crewai.utilities.tool_utils import aexecute_tool_and_check_finality


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."

    def _run(self, term: str = "default") -> str:
        return f"value:{term}"


class _StubLLM(BaseLLM):
    """Never called: the tool path under test does not reach the model."""

    def __init__(self) -> None:
        super().__init__(model="stub-model")

    def call(self, *args, **kwargs) -> str:
        raise AssertionError("the model should not be called")

    def supports_function_calling(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def clean_hooks():
    clear_all()
    yield
    crewai_event_bus.flush()
    clear_all()


def _react_text_executor(
    tool_input: str = '{"term": "hello"}',
) -> tuple[AgentExecutor, AgentAction]:
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
        tool_input=tool_input,
        text=(
            "Thought: I should look this up\n"
            f"Action: lookup\nAction Input: {tool_input}"
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


@pytest.mark.parametrize(
    "tool_input", ['{"term": "original"}', "{}"], ids=["original", "empty"]
)
def test_pre_tool_hook_edits_reach_the_tool_on_react_text_path(tool_input):
    """In-place tool_input edits apply even when the call has no arguments."""

    @on(InterceptionPoint.PRE_TOOL_CALL)
    def rewrite(context):
        context.tool_input["term"] = "effective"

    executor, action = _react_text_executor(tool_input)
    executor.agent = Agent(
        role="Researcher", goal="Look things up", backstory="Curious.", llm=_StubLLM()
    )

    executor.execute_tool_action()

    assert action.result == "value:effective"


@pytest.mark.asyncio
async def test_pre_tool_hook_edits_reach_the_tool_on_async_react_path():
    @on(InterceptionPoint.PRE_TOOL_CALL)
    def rewrite(context):
        context.tool_input["term"] = "effective"

    tool = LookupTool()
    action = AgentAction(
        thought="I should look this up",
        tool="lookup",
        tool_input="{}",
        text="Thought: I should look this up\nAction: lookup\nAction Input: {}",
    )

    result = await aexecute_tool_and_check_finality(
        agent_action=action,
        tools=to_langchain([tool]),
        tools_handler=ToolsHandler(cache=CacheHandler()),
        agent=Agent(
            role="Researcher",
            goal="Look things up",
            backstory="Curious.",
            llm=_StubLLM(),
        ),
    )

    assert result.result == "value:effective"


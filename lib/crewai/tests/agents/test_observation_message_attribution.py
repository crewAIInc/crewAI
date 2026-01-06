"""Tests for proper message attribution to prevent LLM observation hallucination.

This module tests that tool observations are correctly attributed to user messages
rather than assistant messages, which prevents the LLM from learning to hallucinate
fake observations during tool calls.

Related to GitHub issue #4181.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.tools.tool_types import ToolResult
from crewai.utilities.agent_utils import handle_agent_action_core


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"
    return agent


@pytest.fixture
def mock_task() -> MagicMock:
    """Create a mock task for testing."""
    task = MagicMock()
    task.description = "Test task description"
    return task


@pytest.fixture
def mock_crew() -> MagicMock:
    """Create a mock crew for testing."""
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    return crew


@pytest.fixture
def mock_tools_handler() -> MagicMock:
    """Create a mock tools handler."""
    return MagicMock()


@pytest.fixture
def executor(
    mock_llm: MagicMock,
    mock_agent: MagicMock,
    mock_task: MagicMock,
    mock_crew: MagicMock,
    mock_tools_handler: MagicMock,
) -> CrewAgentExecutor:
    """Create a CrewAgentExecutor instance for testing."""
    return CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=mock_crew,
        agent=mock_agent,
        prompt={"prompt": "Test prompt {input} {tool_names} {tools}"},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=["Observation:"],
        tools_description="",
        tools_handler=mock_tools_handler,
    )


class TestHandleAgentActionCore:
    """Tests for handle_agent_action_core function."""

    def test_stores_llm_response_before_observation(self) -> None:
        """Test that llm_response is stored before observation is appended."""
        original_text = "Thought: I need to search\nAction: search\nAction Input: query"
        action = AgentAction(
            thought="I need to search",
            tool="search",
            tool_input="query",
            text=original_text,
        )
        tool_result = ToolResult(result="Search result: found data", result_as_answer=False)

        result = handle_agent_action_core(
            formatted_answer=action,
            tool_result=tool_result,
        )

        assert isinstance(result, AgentAction)
        assert result.llm_response == original_text
        assert "Observation:" in result.text
        assert result.result == "Search result: found data"

    def test_text_contains_observation_for_logging(self) -> None:
        """Test that text contains observation for logging purposes."""
        action = AgentAction(
            thought="Testing",
            tool="test_tool",
            tool_input="{}",
            text="Thought: Testing\nAction: test_tool\nAction Input: {}",
        )
        tool_result = ToolResult(result="Tool output", result_as_answer=False)

        result = handle_agent_action_core(
            formatted_answer=action,
            tool_result=tool_result,
        )

        assert isinstance(result, AgentAction)
        assert "Observation: Tool output" in result.text

    def test_result_as_answer_returns_agent_finish(self) -> None:
        """Test that result_as_answer=True returns AgentFinish."""
        action = AgentAction(
            thought="Using tool",
            tool="final_tool",
            tool_input="{}",
            text="Thought: Using tool\nAction: final_tool\nAction Input: {}",
        )
        tool_result = ToolResult(result="Final answer from tool", result_as_answer=True)

        result = handle_agent_action_core(
            formatted_answer=action,
            tool_result=tool_result,
        )

        assert isinstance(result, AgentFinish)
        assert result.output == "Final answer from tool"


class TestCrewAgentExecutorMessageAttribution:
    """Tests for proper message attribution in CrewAgentExecutor."""

    def test_tool_observation_not_in_assistant_message(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that tool observations are not attributed to assistant messages.

        This is the core fix for GitHub issue #4181 - observations should be
        in user messages, not assistant messages, to prevent LLM hallucination.
        """
        call_count = 0

        def mock_llm_response(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: I need to use a tool\n"
                    "Action: test_tool\n"
                    'Action Input: {"arg": "value"}'
                )
            return "Thought: I have the answer\nFinal Answer: Done"

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            side_effect=mock_llm_response,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.execute_tool_and_check_finality",
                return_value=ToolResult(
                    result="Tool executed successfully", result_as_answer=False
                ),
            ):
                with patch.object(executor, "_show_logs"):
                    result = executor._invoke_loop()

        assert isinstance(result, AgentFinish)

        assistant_messages = [
            msg for msg in executor.messages if msg.get("role") == "assistant"
        ]
        user_messages = [msg for msg in executor.messages if msg.get("role") == "user"]

        for msg in assistant_messages:
            content = msg.get("content", "")
            assert "Observation:" not in content, (
                f"Assistant message should not contain 'Observation:'. "
                f"Found: {content[:100]}..."
            )

        observation_in_user = any(
            "Observation:" in msg.get("content", "") for msg in user_messages
        )
        assert observation_in_user, (
            "Tool observation should be in a user message, not assistant message"
        )

    def test_llm_response_in_assistant_message(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that the LLM's actual response is in assistant messages."""
        call_count = 0
        llm_action_text = (
            "Thought: I need to use a tool\n"
            "Action: test_tool\n"
            'Action Input: {"arg": "value"}'
        )

        def mock_llm_response(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return llm_action_text
            return "Thought: I have the answer\nFinal Answer: Done"

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            side_effect=mock_llm_response,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.execute_tool_and_check_finality",
                return_value=ToolResult(
                    result="Tool executed successfully", result_as_answer=False
                ),
            ):
                with patch.object(executor, "_show_logs"):
                    executor._invoke_loop()

        assistant_messages = [
            msg for msg in executor.messages if msg.get("role") == "assistant"
        ]

        llm_response_found = any(
            "Action: test_tool" in msg.get("content", "") for msg in assistant_messages
        )
        assert llm_response_found, (
            "LLM's action response should be in an assistant message"
        )

    def test_message_order_after_tool_use(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that messages are in correct order: assistant (action) then user (observation)."""
        call_count = 0

        def mock_llm_response(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: I need to use a tool\n"
                    "Action: test_tool\n"
                    'Action Input: {"arg": "value"}'
                )
            return "Thought: I have the answer\nFinal Answer: Done"

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            side_effect=mock_llm_response,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.execute_tool_and_check_finality",
                return_value=ToolResult(
                    result="Tool executed successfully", result_as_answer=False
                ),
            ):
                with patch.object(executor, "_show_logs"):
                    executor._invoke_loop()

        action_msg_idx = None
        observation_msg_idx = None

        for i, msg in enumerate(executor.messages):
            content = msg.get("content", "")
            if "Action: test_tool" in content and msg.get("role") == "assistant":
                action_msg_idx = i
            if "Observation:" in content and msg.get("role") == "user":
                observation_msg_idx = i

        assert action_msg_idx is not None, "Action message not found"
        assert observation_msg_idx is not None, "Observation message not found"
        assert observation_msg_idx == action_msg_idx + 1, (
            f"Observation (user) should immediately follow action (assistant). "
            f"Action at {action_msg_idx}, Observation at {observation_msg_idx}"
        )


class TestAgentActionLlmResponseField:
    """Tests for the llm_response field on AgentAction."""

    def test_agent_action_has_llm_response_field(self) -> None:
        """Test that AgentAction has llm_response field."""
        action = AgentAction(
            thought="Test",
            tool="test",
            tool_input="{}",
            text="Test text",
        )
        assert hasattr(action, "llm_response")
        assert action.llm_response is None

    def test_agent_action_llm_response_can_be_set(self) -> None:
        """Test that llm_response can be set on AgentAction."""
        action = AgentAction(
            thought="Test",
            tool="test",
            tool_input="{}",
            text="Test text",
        )
        action.llm_response = "Original LLM response"
        assert action.llm_response == "Original LLM response"

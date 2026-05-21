from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.llm import LLM
from crewai.utilities.token_counter_callback import TokenCalcHandler


@pytest.mark.vcr(filter_headers=["authorization"])
def test_llm_callback_replacement():
    llm = LLM(model="gpt-4o-mini")

    calc_handler_1 = TokenCalcHandler(token_cost_process=TokenProcess())
    calc_handler_2 = TokenCalcHandler(token_cost_process=TokenProcess())

    llm.call(
        messages=[{"role": "user", "content": "Hello, world!"}],
        callbacks=[calc_handler_1],
    )
    usage_metrics_1 = calc_handler_1.token_cost_process.get_summary()

    llm.call(
        messages=[{"role": "user", "content": "Hello, world from another agent!"}],
        callbacks=[calc_handler_2],
    )
    usage_metrics_2 = calc_handler_2.token_cost_process.get_summary()

    # The first handler should not have been updated
    assert usage_metrics_1.successful_requests == 1
    assert usage_metrics_2.successful_requests == 1
    assert usage_metrics_1 == calc_handler_1.token_cost_process.get_summary()


def _make_response(content, reasoning_content=None):
    """Build a litellm ModelResponse, optionally with reasoning_content."""
    msg_kwargs = {"content": content, "role": "assistant"}
    if reasoning_content is not None:
        msg_kwargs["reasoning_content"] = reasoning_content
    message = Message(**msg_kwargs)
    choice = Choices(message=message, index=0, finish_reason="stop")
    return ModelResponse(choices=[choice])


@patch("crewai.llm.litellm.completion")
def test_llm_call_stores_reasoning_content(mock_completion):
    """LLM.call should store reasoning_content from the response."""
    mock_completion.return_value = _make_response(
        content="Paris",
        reasoning_content="The user asked about the capital of France.",
    )
    llm = LLM(model="deepseek/deepseek-reasoner")
    result = llm.call([{"role": "user", "content": "What is the capital of France?"}])

    assert result == "Paris"
    assert llm.reasoning_content == "The user asked about the capital of France."


@patch("crewai.llm.litellm.completion")
def test_llm_call_no_reasoning_content(mock_completion):
    """LLM.call should set reasoning_content to None when absent."""
    mock_completion.return_value = _make_response(content="Hello!")
    llm = LLM(model="gpt-4o")
    result = llm.call([{"role": "user", "content": "Hi"}])

    assert result == "Hello!"
    assert llm.reasoning_content is None


@patch("crewai.llm.litellm.completion")
def test_llm_call_reasoning_content_reset_between_calls(mock_completion):
    """reasoning_content should be reset on each call."""
    mock_completion.return_value = _make_response(
        content="first", reasoning_content="thinking1"
    )
    llm = LLM(model="deepseek/deepseek-reasoner")
    llm.call([{"role": "user", "content": "q1"}])
    assert llm.reasoning_content == "thinking1"

    # Second call without reasoning_content
    mock_completion.return_value = _make_response(content="second")
    llm.call([{"role": "user", "content": "q2"}])
    assert llm.reasoning_content is None


class TestExecutorReasoningContent:
    """Tests for reasoning_content propagation in CrewAgentExecutor."""

    def _build_executor(self, llm):
        """Build a minimal CrewAgentExecutor for testing."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor
        from crewai.agents.tools_handler import ToolsHandler

        agent = MagicMock()
        agent.role = "test"
        agent.verbose = False
        agent.id = "agent-1"

        task = MagicMock()
        task.description = "test task"

        crew = MagicMock()
        crew.verbose = False
        crew._train = False

        tools_handler = ToolsHandler()

        executor = CrewAgentExecutor(
            llm=llm,
            task=task,
            crew=crew,
            agent=agent,
            prompt={"system": "You are helpful.", "user": "{input}{tool_names}{tools}"},
            max_iter=3,
            tools=[],
            tools_names="",
            stop_words=["Observation:"],
            tools_description="",
            tools_handler=tools_handler,
        )
        return executor

    def test_format_msg_includes_reasoning_content(self):
        """_format_msg should include reasoning_content for assistant messages."""
        llm = MagicMock()
        llm.supports_stop_words.return_value = True
        llm.stop = None
        executor = self._build_executor(llm)

        msg = executor._format_msg(
            "Hello", role="assistant", reasoning_content="thinking..."
        )
        assert msg == {
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "thinking...",
        }

    def test_format_msg_omits_reasoning_content_for_user(self):
        """_format_msg should not include reasoning_content for user messages."""
        llm = MagicMock()
        llm.supports_stop_words.return_value = True
        llm.stop = None
        executor = self._build_executor(llm)

        msg = executor._format_msg(
            "Hello", role="user", reasoning_content="thinking..."
        )
        assert msg == {"role": "user", "content": "Hello"}

    def test_format_msg_omits_reasoning_content_when_none(self):
        """_format_msg should not include reasoning_content key when it is None."""
        llm = MagicMock()
        llm.supports_stop_words.return_value = True
        llm.stop = None
        executor = self._build_executor(llm)

        msg = executor._format_msg("Hello", role="assistant", reasoning_content=None)
        assert msg == {"role": "assistant", "content": "Hello"}
        assert "reasoning_content" not in msg

    @patch("crewai.llm.litellm.completion")
    def test_invoke_loop_preserves_reasoning_content_in_messages(
        self, mock_completion
    ):
        """The invoke loop should include reasoning_content in assistant messages."""
        llm = LLM(model="deepseek/deepseek-reasoner")

        # First call returns an intermediate response (not a final answer)
        # Second call returns the final answer
        mock_completion.side_effect = [
            _make_response(
                content="Thought: I need to think about this.\nFinal Answer: 42",
                reasoning_content="Let me reason step by step...",
            ),
        ]

        executor = self._build_executor(llm)
        executor.invoke(
            {"input": "What is the answer?", "tool_names": "", "tools": ""}
        )

        # Find assistant messages in the message history
        assistant_msgs = [
            m for m in executor.messages if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].get("reasoning_content") == "Let me reason step by step..."

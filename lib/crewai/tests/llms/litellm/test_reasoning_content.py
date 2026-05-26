"""Tests for reasoning_content support (DeepSeek thinking mode).

Verifies that reasoning_content from LLM responses is:
1. Extracted and stored by LLM.call()
2. Propagated into assistant messages by the executor
3. Omitted when the model does not return it
"""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from crewai.llm import LLM
from crewai.utilities.agent_utils import format_message_for_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(content: str, reasoning_content: str | None = None) -> ModelResponse:
    """Build a litellm ModelResponse, optionally with reasoning_content."""
    msg_kwargs: dict[str, Any] = {"content": content, "role": "assistant"}
    if reasoning_content is not None:
        msg_kwargs["reasoning_content"] = reasoning_content
    message = Message(**msg_kwargs)
    choice = Choices(message=message, index=0, finish_reason="stop")
    return ModelResponse(choices=[choice])


# ---------------------------------------------------------------------------
# LLM.call tests
# ---------------------------------------------------------------------------

class TestLLMReasoningContent:
    """LLM.call should extract and store reasoning_content."""

    @patch("crewai.llm.litellm.completion")
    def test_stores_reasoning_content(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _make_response(
            content="Paris",
            reasoning_content="The user asked about the capital of France.",
        )
        llm = LLM(model="deepseek/deepseek-reasoner", is_litellm=True)
        result = llm.call(
            [{"role": "user", "content": "What is the capital of France?"}]
        )

        assert result == "Paris"
        assert llm.reasoning_content == "The user asked about the capital of France."

    @patch("crewai.llm.litellm.completion")
    def test_none_when_absent(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _make_response(content="Hello!")
        llm = LLM(model="gpt-4o", is_litellm=True)
        result = llm.call([{"role": "user", "content": "Hi"}])

        assert result == "Hello!"
        assert llm.reasoning_content is None

    @patch("crewai.llm.litellm.completion")
    def test_resets_between_calls(self, mock_completion: MagicMock) -> None:
        mock_completion.return_value = _make_response(
            content="first", reasoning_content="thinking1"
        )
        llm = LLM(model="deepseek/deepseek-reasoner", is_litellm=True)
        llm.call([{"role": "user", "content": "q1"}])
        assert llm.reasoning_content == "thinking1"

        mock_completion.return_value = _make_response(content="second")
        llm.call([{"role": "user", "content": "q2"}])
        assert llm.reasoning_content is None


# ---------------------------------------------------------------------------
# format_message_for_llm tests
# ---------------------------------------------------------------------------

class TestFormatMessageReasoningContent:
    """format_message_for_llm should handle reasoning_content correctly."""

    def test_includes_reasoning_content_for_assistant(self) -> None:
        msg = format_message_for_llm(
            "Hello", role="assistant", reasoning_content="thinking..."
        )
        assert msg == {
            "role": "assistant",
            "content": "Hello",
            "reasoning_content": "thinking...",
        }

    def test_omits_reasoning_content_for_user(self) -> None:
        msg = format_message_for_llm(
            "Hello", role="user", reasoning_content="thinking..."
        )
        assert msg == {"role": "user", "content": "Hello"}

    def test_omits_reasoning_content_when_none(self) -> None:
        msg = format_message_for_llm(
            "Hello", role="assistant", reasoning_content=None
        )
        assert msg == {"role": "assistant", "content": "Hello"}
        assert "reasoning_content" not in msg

    def test_omits_reasoning_content_when_empty_string(self) -> None:
        msg = format_message_for_llm(
            "Hello", role="assistant", reasoning_content=""
        )
        assert msg == {"role": "assistant", "content": "Hello"}
        assert "reasoning_content" not in msg


# ---------------------------------------------------------------------------
# CrewAgentExecutor unit tests
# ---------------------------------------------------------------------------

def _build_crew_executor(llm: Any) -> Any:
    """Build a minimal CrewAgentExecutor using model_construct to skip validation."""
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.utilities.agent_utils import format_message_for_llm

    agent = MagicMock()
    agent.role = "test"
    agent.verbose = False
    agent.id = "agent-1"
    agent.key = "agent-key"
    agent.security_config = MagicMock()

    task = MagicMock()
    task.name = "test task"
    task.description = "test task"
    task.id = "task-1"

    crew = MagicMock()
    crew.verbose = False
    crew._train = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        executor = CrewAgentExecutor.model_construct(
            llm=llm,
            task=task,
            crew=crew,
            agent=agent,
            messages=[
                format_message_for_llm("You are helpful.", role="system"),
            ],
            iterations=0,
            max_iter=3,
            tools=[],
            original_tools=[],
            tools_names="",
            stop=[],
            tools_description="",
            tools_handler=None,
            step_callback=None,
            function_calling_llm=None,
            respect_context_window=False,
            request_within_rpm_limit=None,
            callbacks=[],
            response_model=None,
            ask_for_human_input=False,
            log_error_after=3,
            before_llm_call_hooks=[],
            after_llm_call_hooks=[],
        )
    return executor


class TestCrewExecutorReasoningContent:
    """CrewAgentExecutor should propagate reasoning_content to message history."""

    def test_get_llm_reasoning_content(self) -> None:
        llm = MagicMock()
        llm.reasoning_content = "some reasoning"
        executor = _build_crew_executor(llm)
        assert executor._get_llm_reasoning_content() == "some reasoning"

    def test_get_llm_reasoning_content_missing(self) -> None:
        llm = MagicMock(
            spec=["call", "supports_stop_words", "supports_function_calling", "stop"]
        )
        executor = _build_crew_executor(llm)
        assert executor._get_llm_reasoning_content() is None

    def test_append_message_includes_reasoning_content(self) -> None:
        llm = MagicMock()
        executor = _build_crew_executor(llm)
        initial_count = len(executor.messages)

        executor._append_message(
            "hello", role="assistant", reasoning_content="thinking..."
        )

        new_msg = executor.messages[initial_count]
        assert new_msg["role"] == "assistant"
        assert new_msg["content"] == "hello"
        assert new_msg["reasoning_content"] == "thinking..."

    def test_append_message_omits_reasoning_content_when_none(self) -> None:
        llm = MagicMock()
        executor = _build_crew_executor(llm)
        initial_count = len(executor.messages)

        executor._append_message("hello", role="assistant", reasoning_content=None)

        new_msg = executor.messages[initial_count]
        assert new_msg["role"] == "assistant"
        assert new_msg["content"] == "hello"
        assert "reasoning_content" not in new_msg

    @patch("crewai.llm.litellm.completion")
    def test_invoke_loop_preserves_reasoning_content(
        self, mock_completion: MagicMock
    ) -> None:
        """The ReAct invoke loop should include reasoning_content in assistant messages."""
        llm = LLM(model="deepseek/deepseek-reasoner", is_litellm=True)

        mock_completion.return_value = _make_response(
            content="Thought: I need to think about this.\nFinal Answer: 42",
            reasoning_content="Let me reason step by step...",
        )

        executor = _build_crew_executor(llm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            executor.invoke(
                {"input": "What is the answer?", "tool_names": "", "tools": ""}
            )

        assistant_msgs = [
            m for m in executor.messages if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) >= 1
        assert (
            assistant_msgs[0].get("reasoning_content")
            == "Let me reason step by step..."
        )

    @patch("crewai.llm.litellm.completion")
    def test_invoke_loop_no_reasoning_content_for_normal_models(
        self, mock_completion: MagicMock
    ) -> None:
        """Assistant messages should NOT have reasoning_content for normal models."""
        llm = LLM(model="gpt-4o", is_litellm=True)

        mock_completion.return_value = _make_response(
            content="Thought: Simple question.\nFinal Answer: Hello!",
        )

        executor = _build_crew_executor(llm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            executor.invoke(
                {"input": "Say hi", "tool_names": "", "tools": ""}
            )

        assistant_msgs = [
            m for m in executor.messages if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) >= 1
        assert "reasoning_content" not in assistant_msgs[0]

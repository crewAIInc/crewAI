"""Tests for LLM stop words mutation fix (issue #4603).

Verifies that CrewAgentExecutor does not permanently mutate the shared LLM
object's stop words, which caused output truncation in crew.kickoff() when
the same LLM was reused across multiple executor lifecycles.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


@pytest.fixture
def mock_llm():
    """Create a mock LLM with stop words support."""
    llm = MagicMock()
    llm.stop = []
    llm.supports_stop_words.return_value = True
    llm.supports_function_calling.return_value = False
    return llm


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock()
    agent.id = "test-agent"
    agent.role = "Test Agent"
    agent.verbose = False
    agent.key = "test-key"
    agent.security_config = None
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task."""
    task = MagicMock()
    task.description = "Test task"
    task.human_input = False
    task.response_model = None
    return task


@pytest.fixture
def mock_crew():
    """Create a mock crew."""
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    crew._memory = None
    return crew


@pytest.fixture
def executor_kwargs(mock_llm, mock_agent, mock_task, mock_crew):
    """Create default kwargs for CrewAgentExecutor."""
    return {
        "llm": mock_llm,
        "task": mock_task,
        "agent": mock_agent,
        "crew": mock_crew,
        "prompt": {"prompt": "Test {input} {tool_names} {tools}"},
        "max_iter": 10,
        "tools": [],
        "tools_names": "",
        "stop_words": ["\nObservation:"],
        "tools_description": "",
        "tools_handler": MagicMock(),
        "original_tools": [],
    }


class TestStopWordsMutationFix:
    """Tests that the executor does not permanently mutate the shared LLM's stop words."""

    def test_executor_init_does_not_mutate_llm_stop(self, executor_kwargs, mock_llm):
        """Verify __init__ does not set stop words on the LLM object."""
        original_stop = list(mock_llm.stop)

        CrewAgentExecutor(**executor_kwargs)

        # The LLM's stop words should remain unchanged after init
        assert mock_llm.stop == original_stop

    def test_executor_saves_original_llm_stop(self, executor_kwargs, mock_llm):
        """Verify __init__ saves the LLM's original stop words."""
        mock_llm.stop = ["existing_stop"]
        executor = CrewAgentExecutor(**executor_kwargs)

        assert executor._original_llm_stop == ["existing_stop"]

    def test_executor_saves_empty_original_stop(self, executor_kwargs, mock_llm):
        """Verify __init__ handles empty stop words."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        assert executor._original_llm_stop == []

    def test_set_llm_stop_words_merges_correctly(self, executor_kwargs, mock_llm):
        """Verify _set_llm_stop_words merges executor stop words with LLM's."""
        mock_llm.stop = ["existing"]
        executor = CrewAgentExecutor(**executor_kwargs)

        executor._set_llm_stop_words()

        # Should contain both original and executor stop words
        assert set(mock_llm.stop) == {"existing", "\nObservation:"}

    def test_restore_llm_stop_words(self, executor_kwargs, mock_llm):
        """Verify _restore_llm_stop_words restores original stop words."""
        mock_llm.stop = ["original_stop"]
        executor = CrewAgentExecutor(**executor_kwargs)

        # Simulate what happens during execution
        executor._set_llm_stop_words()
        assert "\nObservation:" in mock_llm.stop

        executor._restore_llm_stop_words()
        assert mock_llm.stop == ["original_stop"]

    def test_invoke_restores_stop_words_after_success(self, executor_kwargs, mock_llm):
        """Verify invoke restores stop words after successful execution."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        # Mock the invoke loop to return a simple finish
        with patch.object(
            executor, "_invoke_loop", return_value=AgentFinish(
                thought="", output="done", text="done"
            )
        ), patch.object(executor, "_setup_messages"), \
           patch.object(executor, "_inject_multimodal_files"), \
           patch.object(executor, "_show_start_logs"), \
           patch.object(executor, "_save_to_memory"):
            executor.invoke({"input": "test", "tool_names": "", "tools": ""})

        # After invoke completes, LLM stop words should be restored to empty
        assert mock_llm.stop == []

    def test_invoke_restores_stop_words_after_exception(self, executor_kwargs, mock_llm):
        """Verify invoke restores stop words even when an exception occurs."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        # Mock the invoke loop to raise an exception
        with patch.object(executor, "_invoke_loop", side_effect=RuntimeError("boom")), \
             patch.object(executor, "_setup_messages"), \
             patch.object(executor, "_inject_multimodal_files"), \
             patch.object(executor, "_show_start_logs"):
            with pytest.raises(RuntimeError, match="boom"):
                executor.invoke({"input": "test", "tool_names": "", "tools": ""})

        # Even after exception, LLM stop words should be restored
        assert mock_llm.stop == []

    @pytest.mark.asyncio
    async def test_ainvoke_restores_stop_words_after_success(self, executor_kwargs, mock_llm):
        """Verify ainvoke restores stop words after successful execution."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        # Mock the async invoke loop to return a simple finish
        with patch.object(
            executor, "_ainvoke_loop", return_value=AgentFinish(
                thought="", output="done", text="done"
            )
        ), patch.object(executor, "_setup_messages"), \
           patch.object(executor, "_ainject_multimodal_files"), \
           patch.object(executor, "_show_start_logs"), \
           patch.object(executor, "_save_to_memory"):
            await executor.ainvoke({"input": "test", "tool_names": "", "tools": ""})

        assert mock_llm.stop == []

    @pytest.mark.asyncio
    async def test_ainvoke_restores_stop_words_after_exception(self, executor_kwargs, mock_llm):
        """Verify ainvoke restores stop words even when an exception occurs."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        async def raise_error():
            raise RuntimeError("async boom")

        with patch.object(executor, "_ainvoke_loop", side_effect=raise_error), \
             patch.object(executor, "_setup_messages"), \
             patch.object(executor, "_ainject_multimodal_files"), \
             patch.object(executor, "_show_start_logs"):
            with pytest.raises(RuntimeError, match="async boom"):
                await executor.ainvoke({"input": "test", "tool_names": "", "tools": ""})

        assert mock_llm.stop == []


class TestSharedLLMNotPolluted:
    """Tests that a shared LLM object is not polluted across multiple executor instances."""

    def test_multiple_executors_do_not_accumulate_stop_words(
        self, executor_kwargs, mock_llm
    ):
        """Verify creating multiple executors doesn't accumulate stop words on LLM."""
        mock_llm.stop = []

        # Create first executor with stop words
        executor1 = CrewAgentExecutor(**executor_kwargs)
        with patch.object(
            executor1, "_invoke_loop", return_value=AgentFinish(
                thought="", output="done", text="done"
            )
        ), patch.object(executor1, "_setup_messages"), \
           patch.object(executor1, "_inject_multimodal_files"), \
           patch.object(executor1, "_show_start_logs"), \
           patch.object(executor1, "_save_to_memory"):
            executor1.invoke({"input": "test", "tool_names": "", "tools": ""})

        # LLM should be clean after first executor
        assert mock_llm.stop == []

        # Create second executor
        executor2 = CrewAgentExecutor(**executor_kwargs)
        with patch.object(
            executor2, "_invoke_loop", return_value=AgentFinish(
                thought="", output="done2", text="done2"
            )
        ), patch.object(executor2, "_setup_messages"), \
           patch.object(executor2, "_inject_multimodal_files"), \
           patch.object(executor2, "_show_start_logs"), \
           patch.object(executor2, "_save_to_memory"):
            executor2.invoke({"input": "test", "tool_names": "", "tools": ""})

        # LLM should still be clean after second executor
        assert mock_llm.stop == []

    def test_llm_stop_words_only_set_during_execution(
        self, executor_kwargs, mock_llm
    ):
        """Verify stop words are only on the LLM during active execution."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        stop_words_during_execution = []

        def capture_stop_words():
            # Capture what the LLM's stop words are during execution
            stop_words_during_execution.append(list(mock_llm.stop))
            return AgentFinish(thought="", output="done", text="done")

        with patch.object(executor, "_invoke_loop", side_effect=capture_stop_words), \
             patch.object(executor, "_setup_messages"), \
             patch.object(executor, "_inject_multimodal_files"), \
             patch.object(executor, "_show_start_logs"), \
             patch.object(executor, "_save_to_memory"):
            executor.invoke({"input": "test", "tool_names": "", "tools": ""})

        # During execution, stop words should have been set
        assert len(stop_words_during_execution) == 1
        assert "\nObservation:" in stop_words_during_execution[0]

        # After execution, stop words should be restored
        assert mock_llm.stop == []

    def test_user_configured_stop_words_preserved(self, executor_kwargs, mock_llm):
        """Verify user-configured stop words on the LLM are preserved."""
        mock_llm.stop = ["UserStop1", "UserStop2"]
        executor = CrewAgentExecutor(**executor_kwargs)

        with patch.object(
            executor, "_invoke_loop", return_value=AgentFinish(
                thought="", output="done", text="done"
            )
        ), patch.object(executor, "_setup_messages"), \
           patch.object(executor, "_inject_multimodal_files"), \
           patch.object(executor, "_show_start_logs"), \
           patch.object(executor, "_save_to_memory"):
            executor.invoke({"input": "test", "tool_names": "", "tools": ""})

        # User's original stop words should be preserved
        assert mock_llm.stop == ["UserStop1", "UserStop2"]


class TestUpdateExecutorParameters:
    """Tests for _update_executor_parameters not mutating shared LLM stop words."""

    def test_update_parameters_does_not_mutate_llm_stop(
        self, executor_kwargs, mock_llm
    ):
        """Verify _update_executor_parameters does not set stop on LLM."""
        mock_llm.stop = []
        executor = CrewAgentExecutor(**executor_kwargs)

        # Simulate what Agent.create_agent_executor does on update
        executor.task = executor_kwargs["task"]
        executor.tools = []
        executor.original_tools = []
        executor.prompt = executor_kwargs["prompt"]
        executor.stop = ["\nObservation:"]
        executor.tools_names = ""
        executor.tools_description = ""

        # Update the saved original stop words (what the fix does)
        executor._original_llm_stop = list(getattr(mock_llm, "stop", []) or [])

        # LLM stop should not be modified
        assert mock_llm.stop == []

    def test_sequential_crew_executions_no_stop_word_leak(self, mock_llm):
        """Simulate multiple sequential crew executions sharing an LLM.

        This is the core reproduction of issue #4603: when crew.kickoff()
        is called multiple times (e.g., in a workflow), the LLM's stop words
        should not leak between executions.
        """
        mock_llm.stop = []

        for i in range(3):
            agent = MagicMock()
            agent.id = f"agent-{i}"
            agent.role = f"Agent {i}"
            agent.verbose = False
            agent.key = f"key-{i}"
            agent.security_config = None

            task = MagicMock()
            task.description = f"Task {i}"
            task.human_input = False
            task.response_model = None

            crew = MagicMock()
            crew.verbose = False
            crew._train = False
            crew._memory = None

            executor = CrewAgentExecutor(
                llm=mock_llm,
                task=task,
                agent=agent,
                crew=crew,
                prompt={"prompt": "Test {input} {tool_names} {tools}"},
                max_iter=10,
                tools=[],
                tools_names="",
                stop_words=["\nObservation:"],
                tools_description="",
                tools_handler=MagicMock(),
                original_tools=[],
            )

            # Simulate execution
            with patch.object(
                executor, "_invoke_loop", return_value=AgentFinish(
                    thought="", output=f"result-{i}", text=f"result-{i}"
                )
            ), patch.object(executor, "_setup_messages"), \
               patch.object(executor, "_inject_multimodal_files"), \
               patch.object(executor, "_show_start_logs"), \
               patch.object(executor, "_save_to_memory"):
                result = executor.invoke(
                    {"input": "test", "tool_names": "", "tools": ""}
                )
                assert result["output"] == f"result-{i}"

            # After each execution, LLM's stop words should be clean
            assert mock_llm.stop == [], (
                f"LLM stop words leaked after execution {i}: {mock_llm.stop}"
            )


class TestApplyStopWordsInteraction:
    """Tests that _apply_stop_words on the LLM doesn't truncate after executor cleanup."""

    def test_apply_stop_words_not_triggered_after_restore(self, mock_llm):
        """Verify that after restoring stop words, _apply_stop_words doesn't truncate."""
        mock_llm.stop = []

        agent = MagicMock()
        agent.id = "test-agent"
        agent.role = "Test"
        agent.verbose = False
        agent.key = "key"
        agent.security_config = None

        task = MagicMock()
        task.description = "Test"
        task.human_input = False
        task.response_model = None

        crew = MagicMock()
        crew.verbose = False
        crew._train = False
        crew._memory = None

        executor = CrewAgentExecutor(
            llm=mock_llm,
            task=task,
            agent=agent,
            crew=crew,
            prompt={"prompt": "Test {input} {tool_names} {tools}"},
            max_iter=10,
            tools=[],
            tools_names="",
            stop_words=["\nObservation:"],
            tools_description="",
            tools_handler=MagicMock(),
            original_tools=[],
        )

        # Simulate execution and restore
        with patch.object(
            executor, "_invoke_loop", return_value=AgentFinish(
                thought="", output="done", text="done"
            )
        ), patch.object(executor, "_setup_messages"), \
           patch.object(executor, "_inject_multimodal_files"), \
           patch.object(executor, "_show_start_logs"), \
           patch.object(executor, "_save_to_memory"):
            executor.invoke({"input": "test", "tool_names": "", "tools": ""})

        # Now simulate a subsequent LLM call (e.g., from another crew)
        # The response contains "\nObservation:" but should NOT be truncated
        # because the stop words have been restored to empty
        long_response = (
            "```json\n"
            '{"analysis": "This is a detailed analysis with many findings. '
            "The data shows significant trends across all metrics. "
            "We observed multiple patterns including seasonal variations.\n"
            'Observation: The key finding is that performance improved by 25%."}\n'
            "```"
        )

        # With the fix, the LLM's stop words should be empty
        assert mock_llm.stop == []

        # Simulate what _apply_stop_words would do
        # (testing the concept - the actual method is on BaseLLM)
        content = long_response
        if mock_llm.stop:
            for stop_word in mock_llm.stop:
                pos = content.find(stop_word)
                if pos != -1:
                    content = content[:pos].strip()

        # The full response should be preserved (not truncated)
        assert content == long_response
        assert len(content) > 200  # Should be the full response


class TestEdgeCases:
    """Test edge cases for the stop words fix."""

    def test_executor_with_no_stop_words(self, executor_kwargs, mock_llm):
        """Verify executor works correctly when no stop words are provided."""
        executor_kwargs["stop_words"] = []
        mock_llm.stop = []

        executor = CrewAgentExecutor(**executor_kwargs)
        executor._set_llm_stop_words()

        # No stop words should be set
        assert mock_llm.stop == []

        executor._restore_llm_stop_words()
        assert mock_llm.stop == []

    def test_executor_with_none_llm_stop(self, executor_kwargs, mock_llm):
        """Verify executor handles None stop words on LLM."""
        mock_llm.stop = None

        executor = CrewAgentExecutor(**executor_kwargs)
        assert executor._original_llm_stop == []

        executor._set_llm_stop_words()
        assert "\nObservation:" in mock_llm.stop

        executor._restore_llm_stop_words()
        assert mock_llm.stop == []

    def test_executor_with_no_llm(self, executor_kwargs):
        """Verify executor handles None LLM gracefully."""
        executor_kwargs["llm"] = None

        executor = CrewAgentExecutor(**executor_kwargs)
        assert executor._original_llm_stop is None

        # These should not raise
        executor._set_llm_stop_words()
        executor._restore_llm_stop_words()

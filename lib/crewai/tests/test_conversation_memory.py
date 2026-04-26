"""Tests for conversation-aware memory extraction in Agent.kickoff().

Verifies that the overridden _save_to_memory in AgentExecutor includes
the full conversation history when extracting memories, not just the
task description and final result.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.agents.parser import AgentFinish


class FakeMemory:
    """Minimal memory stub for testing."""

    read_only = False
    root_scope = None

    def __init__(self) -> None:
        self.extracted_content: str = ""
        self.remembered: list[str] = []

    def extract_memories(self, content: str) -> list[str]:
        self.extracted_content = content
        return [f"fact from: {content[:40]}"]

    def remember_many(
        self,
        contents: list[str],
        agent_role: str | None = None,
        root_scope: str | None = None,
    ) -> list[Any]:
        self.remembered.extend(contents)
        return []


class FakeAgent:
    """Minimal agent stub."""

    def __init__(self, role: str = "Test Agent", memory: Any = None) -> None:
        self.role = role
        self.memory = memory
        self._logger = MagicMock()


class FakeTask:
    """Minimal task stub."""

    def __init__(
        self,
        description: str = "Test task",
        expected_output: str = "A result",
    ) -> None:
        self.description = description
        self.expected_output = expected_output


def _build_executor(
    agent: FakeAgent,
    task: FakeTask | None = None,
    messages: list[dict[str, str]] | None = None,
):
    """Build an AgentExecutor with model_construct to skip validation."""
    import threading

    from crewai.experimental.agent_executor import AgentExecutor, AgentExecutorState

    state = AgentExecutorState()
    if messages:
        state.messages = messages  # type: ignore[assignment]

    executor = AgentExecutor.model_construct(
        agent=agent,
        task=task,
        crew=None,
        llm=MagicMock(),
        callbacks=None,
        prompt=None,
        original_tools=[],
        tools=[],
    )
    executor._state = state
    executor._methods = {}
    executor._method_outputs = []
    executor._completed_methods = set()
    executor._fired_or_listeners = set()
    executor._pending_and_listeners = {}
    executor._method_execution_counts = {}
    executor._method_call_counts = {}
    executor._event_futures = []
    executor._human_feedback_method_outputs = {}
    executor._input_history = []
    executor._is_execution_resuming = False
    executor._state_lock = threading.Lock()
    executor._or_listeners_lock = threading.Lock()
    executor._execution_lock = threading.Lock()
    executor._finalize_lock = threading.Lock()
    executor._finalize_called = False
    executor._is_executing = False
    executor._has_been_invoked = False
    executor._last_parser_error = None
    executor._last_context_error = None
    executor._step_executor = None
    executor._planner_observer = None
    return executor


class TestConversationAwareMemory:
    """Tests for the overridden _save_to_memory that includes conversation."""

    def test_conversation_included_in_extraction(self):
        """Memory extraction should include conversation turns."""
        memory = FakeMemory()
        agent = FakeAgent(memory=memory)
        task = FakeTask(description="Find the weather")

        messages = [
            {"role": "user", "content": "What's the weather in SF?"},
            {"role": "assistant", "content": "Let me check..."},
            {"role": "user", "content": "Also check NYC please"},
            {"role": "assistant", "content": "SF: 62F sunny. NYC: 45F cloudy."},
        ]

        executor = _build_executor(agent, task, messages)
        output = AgentFinish(output="SF: 62F sunny. NYC: 45F cloudy.", text="SF: 62F sunny. NYC: 45F cloudy.", thought="")
        executor._save_to_memory(output)

        # The extracted content should contain conversation turns
        assert "user: What's the weather in SF?" in memory.extracted_content
        assert "user: Also check NYC please" in memory.extracted_content
        assert "Conversation:" in memory.extracted_content
        assert "Final result:" in memory.extracted_content
        assert len(memory.remembered) > 0

    def test_task_metadata_included(self):
        """Task description and expected output should still be present."""
        memory = FakeMemory()
        agent = FakeAgent(memory=memory)
        task = FakeTask(description="Analyze sales data", expected_output="A report")

        messages = [{"role": "user", "content": "Run the analysis"}]

        executor = _build_executor(agent, task, messages)
        output = AgentFinish(output="Report ready", text="Report ready", thought="")
        executor._save_to_memory(output)

        assert "Task: Analyze sales data" in memory.extracted_content
        assert "Expected result: A report" in memory.extracted_content

    def test_no_messages_falls_back(self):
        """With no conversation history, falls back to task+result format."""
        memory = FakeMemory()
        agent = FakeAgent(memory=memory)
        task = FakeTask(description="Simple task")

        executor = _build_executor(agent, task, messages=[])
        output = AgentFinish(output="Done", text="Done", thought="")
        executor._save_to_memory(output)

        # Fallback format
        assert "Task: Simple task" in memory.extracted_content
        assert "Result: Done" in memory.extracted_content
        # No "Conversation:" header
        assert "Conversation:" not in memory.extracted_content

    def test_no_memory_is_noop(self):
        """No memory attached should not raise."""
        agent = FakeAgent(memory=None)
        task = FakeTask()

        executor = _build_executor(agent, task)
        output = AgentFinish(output="Done", text="Done", thought="")
        executor._save_to_memory(output)  # should not raise

    def test_read_only_memory_is_noop(self):
        """Read-only memory should not attempt to save."""
        memory = FakeMemory()
        memory.read_only = True
        agent = FakeAgent(memory=memory)
        task = FakeTask()

        executor = _build_executor(agent, task, messages=[{"role": "user", "content": "hello"}])
        output = AgentFinish(output="Done", text="Done", thought="")
        executor._save_to_memory(output)

        assert memory.extracted_content == ""
        assert len(memory.remembered) == 0

    def test_long_conversation_truncated(self):
        """Only the last 20 turns should be included."""
        memory = FakeMemory()
        agent = FakeAgent(memory=memory)
        task = FakeTask()

        messages = [
            {"role": "user", "content": f"message {i}"}
            for i in range(30)
        ]

        executor = _build_executor(agent, task, messages)
        output = AgentFinish(output="Done", text="Done", thought="")
        executor._save_to_memory(output)

        # Should NOT include the first 10 messages
        assert "message 0" not in memory.extracted_content
        assert "message 9" not in memory.extracted_content
        # Should include the last 20
        assert "message 10" in memory.extracted_content
        assert "message 29" in memory.extracted_content

    def test_scoped_memory_saves_to_agent_root(self):
        """When memory has a root_scope, saves should go under /root/agent/<role>."""
        memory = FakeMemory()
        memory.root_scope = "/company"
        agent = FakeAgent(role="Researcher", memory=memory)
        task = FakeTask()

        messages = [{"role": "user", "content": "Research AI trends"}]

        executor = _build_executor(agent, task, messages)
        output = AgentFinish(output="Report", text="Report", thought="")
        executor._save_to_memory(output)

        assert len(memory.remembered) > 0

    def test_no_task_still_works(self):
        """Agent.kickoff() without a task should still extract from conversation."""
        memory = FakeMemory()
        agent = FakeAgent(memory=memory)

        messages = [
            {"role": "user", "content": "Remember that Joe prefers dark mode"},
            {"role": "assistant", "content": "Got it, noted."},
        ]

        executor = _build_executor(agent, task=None, messages=messages)
        output = AgentFinish(output="Noted.", text="Noted.", thought="")
        executor._save_to_memory(output)

        assert "user: Remember that Joe prefers dark mode" in memory.extracted_content

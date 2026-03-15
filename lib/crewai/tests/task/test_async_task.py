"""Tests for async task execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crewai.agent import Agent
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.tasks.output_format import OutputFormat


@pytest.fixture
def test_agent() -> Agent:
    """Create a test agent."""
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm="gpt-4o-mini",
        verbose=False,
    )


class TestAsyncTaskExecution:
    """Tests for async task execution methods."""

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_basic(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test basic async task execution."""
        mock_execute.return_value = "Async task result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        result = await task.aexecute_sync()

        assert result is not None
        assert isinstance(result, TaskOutput)
        assert result.raw == "Async task result"
        assert result.agent == "Test Agent"
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_with_context(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async task execution with context."""
        mock_execute.return_value = "Async result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        context = "Additional context for the task"
        result = await task.aexecute_sync(context=context)

        assert result is not None
        assert task.prompt_context == context
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args[1]
        assert call_kwargs["context"] == context

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_with_tools(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async task execution with custom tools."""
        mock_execute.return_value = "Async result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        result = await task.aexecute_sync(tools=[mock_tool])

        assert result is not None
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args[1]
        assert mock_tool in call_kwargs["tools"]

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_sets_start_and_end_time(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async execution sets start and end times."""
        mock_execute.return_value = "Async result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        assert task.start_time is None
        assert task.end_time is None

        await task.aexecute_sync()

        assert task.start_time is not None
        assert task.end_time is not None
        assert task.end_time >= task.start_time

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_stores_output(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async execution stores the output."""
        mock_execute.return_value = "Async task result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        assert task.output is None

        await task.aexecute_sync()

        assert task.output is not None
        assert task.output.raw == "Async task result"

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_adds_agent_to_processed_by(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async execution adds agent to processed_by_agents."""
        mock_execute.return_value = "Async result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        assert len(task.processed_by_agents) == 0

        await task.aexecute_sync()

        assert "Test Agent" in task.processed_by_agents

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_calls_callback(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async execution calls the callback."""
        mock_execute.return_value = "Async result"
        callback = MagicMock()
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
            callback=callback,
        )

        await task.aexecute_sync()

        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], TaskOutput)

    @pytest.mark.asyncio
    async def test_aexecute_sync_without_agent_raises(self) -> None:
        """Test that async execution without agent raises exception."""
        task = Task(
            description="Test task",
            expected_output="Test output",
        )

        with pytest.raises(Exception) as exc_info:
            await task.aexecute_sync()

        assert "has no agent assigned" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_with_different_agent(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async execution with a different agent than assigned."""
        mock_execute.return_value = "Other agent result"
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        other_agent = Agent(
            role="Other Agent",
            goal="Other goal",
            backstory="Other backstory",
            llm="gpt-4o-mini",
            verbose=False,
        )

        result = await task.aexecute_sync(agent=other_agent)

        assert result.raw == "Other agent result"
        assert result.agent == "Other Agent"
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_handles_exception(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async execution handles exceptions properly."""
        mock_execute.side_effect = RuntimeError("Test error")
        task = Task(
            description="Test task description",
            expected_output="Test expected output",
            agent=test_agent,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await task.aexecute_sync()

        assert "Test error" in str(exc_info.value)
        assert task.end_time is not None


class TestAsyncGuardrails:
    """Tests for async guardrail invocation."""

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_ainvoke_guardrail_success(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async guardrail invocation with successful validation."""
        mock_execute.return_value = "Async task result"

        def guardrail_fn(output: TaskOutput) -> tuple[bool, str]:
            return True, output.raw

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
            guardrail=guardrail_fn,
        )

        result = await task.aexecute_sync()

        assert result is not None
        assert result.raw == "Async task result"

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_ainvoke_guardrail_failure_then_success(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async guardrail that fails then succeeds on retry."""
        mock_execute.side_effect = ["First result", "Second result"]
        call_count = 0

        def guardrail_fn(output: TaskOutput) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return False, "First attempt failed"
            return True, output.raw

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
            guardrail=guardrail_fn,
        )

        result = await task.aexecute_sync()

        assert result is not None
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_ainvoke_guardrail_max_retries_exceeded(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async guardrail that exceeds max retries."""
        mock_execute.return_value = "Async result"

        def guardrail_fn(output: TaskOutput) -> tuple[bool, str]:
            return False, "Always fails"

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
            guardrail=guardrail_fn,
            guardrail_max_retries=2,
        )

        with pytest.raises(Exception) as exc_info:
            await task.aexecute_sync()

        assert "validation after" in str(exc_info.value)
        assert "2 retries" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_ainvoke_multiple_guardrails(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async execution with multiple guardrails."""
        mock_execute.return_value = "Async result"
        guardrail1_called = False
        guardrail2_called = False

        def guardrail1(output: TaskOutput) -> tuple[bool, str]:
            nonlocal guardrail1_called
            guardrail1_called = True
            return True, output.raw

        def guardrail2(output: TaskOutput) -> tuple[bool, str]:
            nonlocal guardrail2_called
            guardrail2_called = True
            return True, output.raw

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
            guardrails=[guardrail1, guardrail2],
        )

        await task.aexecute_sync()

        assert guardrail1_called
        assert guardrail2_called


class TestAsyncTaskOutput:
    """Tests for async task output handling."""

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_output_format_raw(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async execution with raw output format."""
        mock_execute.return_value = '{"key": "value"}'
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
        )

        result = await task.aexecute_sync()

        assert result.output_format == OutputFormat.RAW

    @pytest.mark.asyncio
    @patch("crewai.Agent.aexecute_task", new_callable=AsyncMock)
    async def test_aexecute_sync_task_output_attributes(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that task output has correct attributes."""
        mock_execute.return_value = "Test result"
        task = Task(
            description="Test description",
            expected_output="Test expected",
            agent=test_agent,
            name="Test Task Name",
        )

        result = await task.aexecute_sync()

        assert result.name == "Test Task Name"
        assert result.description == "Test description"
        assert result.expected_output == "Test expected"
        assert result.raw == "Test result"
        assert result.agent == "Test Agent"
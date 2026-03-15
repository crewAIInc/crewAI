"""Tests for async crew execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput


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


@pytest.fixture
def test_task(test_agent: Agent) -> Task:
    """Create a test task."""
    return Task(
        description="Test task description",
        expected_output="Test expected output",
        agent=test_agent,
    )


@pytest.fixture
def test_crew(test_agent: Agent, test_task: Task) -> Crew:
    """Create a test crew."""
    return Crew(
        agents=[test_agent],
        tasks=[test_task],
        verbose=False,
    )


class TestAsyncCrewKickoff:
    """Tests for async crew kickoff methods."""

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_basic(
        self, mock_execute: AsyncMock, test_crew: Crew
    ) -> None:
        """Test basic async crew kickoff."""
        mock_output = TaskOutput(
            description="Test task description",
            raw="Task result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        result = await test_crew.akickoff()

        assert result is not None
        assert isinstance(result, CrewOutput)
        assert result.raw == "Task result"
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_with_inputs(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async crew kickoff with inputs."""
        task = Task(
            description="Test task for {topic}",
            expected_output="Expected output for {topic}",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
        )

        mock_output = TaskOutput(
            description="Test task for AI",
            raw="Task result about AI",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        result = await crew.akickoff(inputs={"topic": "AI"})

        assert result is not None
        assert isinstance(result, CrewOutput)
        mock_execute.assert_called_once()

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_multiple_tasks(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async crew kickoff with multiple tasks."""
        task1 = Task(
            description="First task",
            expected_output="First output",
            agent=test_agent,
        )
        task2 = Task(
            description="Second task",
            expected_output="Second output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task1, task2],
            verbose=False,
        )

        mock_output1 = TaskOutput(
            description="First task",
            raw="First result",
            agent="Test Agent",
        )
        mock_output2 = TaskOutput(
            description="Second task",
            raw="Second result",
            agent="Test Agent",
        )
        mock_execute.side_effect = [mock_output1, mock_output2]

        result = await crew.akickoff()

        assert result is not None
        assert isinstance(result, CrewOutput)
        assert result.raw == "Second result"
        assert mock_execute.call_count == 2

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_handles_exception(
        self, mock_execute: AsyncMock, test_crew: Crew
    ) -> None:
        """Test that async kickoff handles exceptions properly."""
        mock_execute.side_effect = RuntimeError("Test error")

        with pytest.raises(RuntimeError) as exc_info:
            await test_crew.akickoff()

        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_calls_before_callbacks(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async kickoff calls before_kickoff_callbacks."""
        callback_called = False

        def before_callback(inputs: dict | None) -> dict:
            nonlocal callback_called
            callback_called = True
            return inputs or {}

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
            before_kickoff_callbacks=[before_callback],
        )

        mock_output = TaskOutput(
            description="Test task",
            raw="Task result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        await crew.akickoff()

        assert callback_called

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_calls_after_callbacks(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async kickoff calls after_kickoff_callbacks."""
        callback_called = False

        def after_callback(result: CrewOutput) -> CrewOutput:
            nonlocal callback_called
            callback_called = True
            return result

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
            after_kickoff_callbacks=[after_callback],
        )

        mock_output = TaskOutput(
            description="Test task",
            raw="Task result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        await crew.akickoff()

        assert callback_called


class TestAsyncCrewKickoffForEach:
    """Tests for async crew kickoff_for_each methods."""

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_for_each_basic(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test basic async kickoff_for_each."""
        task = Task(
            description="Test task for {topic}",
            expected_output="Expected output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
        )

        mock_output1 = TaskOutput(
            description="Test task for AI",
            raw="Result about AI",
            agent="Test Agent",
        )
        mock_output2 = TaskOutput(
            description="Test task for ML",
            raw="Result about ML",
            agent="Test Agent",
        )
        mock_execute.side_effect = [mock_output1, mock_output2]

        inputs = [{"topic": "AI"}, {"topic": "ML"}]
        results = await crew.akickoff_for_each(inputs)

        assert len(results) == 2
        assert all(isinstance(r, CrewOutput) for r in results)

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_for_each_concurrent(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async kickoff_for_each runs concurrently."""
        task = Task(
            description="Test task for {topic}",
            expected_output="Expected output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
        )

        mock_output = TaskOutput(
            description="Test task",
            raw="Result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        inputs = [{"topic": f"topic_{i}"} for i in range(3)]
        results = await crew.akickoff_for_each(inputs)

        assert len(results) == 3


class TestAsyncTaskExecution:
    """Tests for async task execution within crew."""

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_aexecute_tasks_sequential(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async sequential task execution."""
        task1 = Task(
            description="First task",
            expected_output="First output",
            agent=test_agent,
        )
        task2 = Task(
            description="Second task",
            expected_output="Second output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task1, task2],
            verbose=False,
        )

        mock_output1 = TaskOutput(
            description="First task",
            raw="First result",
            agent="Test Agent",
        )
        mock_output2 = TaskOutput(
            description="Second task",
            raw="Second result",
            agent="Test Agent",
        )
        mock_execute.side_effect = [mock_output1, mock_output2]

        result = await crew._aexecute_tasks(crew.tasks)

        assert result is not None
        assert result.raw == "Second result"
        assert len(result.tasks_output) == 2

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_aexecute_tasks_with_async_task(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test async execution with async_execution task flag."""
        task1 = Task(
            description="Async task",
            expected_output="Async output",
            agent=test_agent,
            async_execution=True,
        )
        task2 = Task(
            description="Sync task",
            expected_output="Sync output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task1, task2],
            verbose=False,
        )

        mock_output1 = TaskOutput(
            description="Async task",
            raw="Async result",
            agent="Test Agent",
        )
        mock_output2 = TaskOutput(
            description="Sync task",
            raw="Sync result",
            agent="Test Agent",
        )
        mock_execute.side_effect = [mock_output1, mock_output2]

        result = await crew._aexecute_tasks(crew.tasks)

        assert result is not None
        assert mock_execute.call_count == 2


class TestAsyncProcessAsyncTasks:
    """Tests for _aprocess_async_tasks method."""

    @pytest.mark.asyncio
    async def test_aprocess_async_tasks_empty(self, test_crew: Crew) -> None:
        """Test processing empty list of async tasks."""
        result = await test_crew._aprocess_async_tasks([])
        assert result == []
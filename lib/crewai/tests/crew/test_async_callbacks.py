"""Tests for async callbacks support in akickoff."""

import asyncio
import pytest
from typing import Any
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


class TestAsyncCallbacksSupport:
    """Tests for async callback support in akickoff."""

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_calls_async_before_callback(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async before_callback is awaited in aprepare_kickoff."""
        callback_result = {"called": False}

        async def async_before_callback(inputs: dict | None) -> dict[str, Any]:
            callback_result["called"] = True
            await asyncio.sleep(0.01)
            return inputs or {}

        task = Task(
            description="Test task for {topic}",
            expected_output="Expected output for {topic}",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
            before_kickoff_callbacks=[async_before_callback],
        )

        mock_output = TaskOutput(
            description="Test task for AI",
            raw="Task result about AI",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        result = await crew.akickoff(inputs={"topic": "AI"})

        assert callback_result["called"], "Async before callback was not called"
        assert result is not None
        assert isinstance(result, CrewOutput)

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_calls_async_after_callback(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that async after_callback is awaited in akickoff."""
        callback_result = {"called": False, "received": None}

        async def async_after_callback(result: CrewOutput) -> CrewOutput:
            nonlocal callback_result
            callback_result["called"] = True
            callback_result["received"] = result
            await asyncio.sleep(0.01)
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
            after_kickoff_callbacks=[async_after_callback],
        )

        mock_output = TaskOutput(
            description="Test task",
            raw="Task result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        result = await crew.akickoff()

        assert callback_result["called"], "Async after callback was not called"
        assert callback_result["received"] is not None
        assert isinstance(callback_result["received"], CrewOutput)

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_mixed_sync_and_async_callbacks(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that mixed sync and async callbacks work together."""
        sync_result = {"called": False}
        async_result = {"called": False, "received": None}

        def sync_before_callback(inputs: dict | None) -> dict:
            sync_result["called"] = True
            return inputs or {}

        async def async_after_callback(result: CrewOutput) -> CrewOutput:
            nonlocal async_result
            async_result["called"] = True
            async_result["received"] = result
            await asyncio.sleep(0.01)
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
            before_kickoff_callbacks=[sync_before_callback],
            after_kickoff_callbacks=[async_after_callback],
        )

        mock_output = TaskOutput(
            description="Test task",
            raw="Task result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        result = await crew.akickoff()

        assert sync_result["called"], "Sync before callback was not called"
        assert async_result["called"], "Async after callback was not called"
        assert async_result["received"] is not None

    @pytest.mark.asyncio
    @patch("crewai.task.Task.aexecute_sync", new_callable=AsyncMock)
    async def test_akickoff_empty_callbacks(
        self, mock_execute: AsyncMock, test_agent: Agent
    ) -> None:
        """Test that empty callbacks list still works normally."""
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=test_agent,
        )
        crew = Crew(
            agents=[test_agent],
            tasks=[task],
            verbose=False,
            before_kickoff_callbacks=[],
            after_kickoff_callbacks=[],
        )

        mock_output = TaskOutput(
            description="Test task",
            raw="Task result",
            agent="Test Agent",
        )
        mock_execute.return_value = mock_output

        result = await crew.akickoff()

        assert result is not None
        assert isinstance(result, CrewOutput)
        assert result.raw == "Task result"

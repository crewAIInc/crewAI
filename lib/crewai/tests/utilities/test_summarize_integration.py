"""
Integration tests for structured context compaction (summarize_messages).
"""

from __future__ import annotations

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.llm import LLM
from crewai.task import Task
class TestCrewKickoffCompaction:
    """Test compaction triggered via Crew.kickoff() with small context window."""

    @pytest.mark.vcr()
    def test_crew_kickoff_compaction_openai(self) -> None:
        """Test that compaction is triggered during kickoff with small context_window_size."""
        llm = LLM(model="gpt-4o-mini", temperature=0)
        # Force a very small context window to trigger compaction
        llm.context_window_size = 500

        agent = Agent(
            role="Researcher",
            goal="Find information about Python programming",
            backstory="You are an expert researcher.",
            llm=llm,
            verbose=False,
            max_iter=2,
        )

        task = Task(
            description="What is Python? Give a brief answer.",
            expected_output="A short description of Python.",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        result = crew.kickoff()
        assert result is not None


class TestAgentExecuteTaskCompaction:
    """Test compaction triggered via Agent.execute_task()."""

    @pytest.mark.vcr()
    def test_agent_execute_task_compaction(self) -> None:
        """Test that Agent.execute_task() works with small context_window_size."""
        llm = LLM(model="gpt-4o-mini", temperature=0)
        llm.context_window_size = 500

        agent = Agent(
            role="Writer",
            goal="Write concise content",
            backstory="You are a skilled writer.",
            llm=llm,
            verbose=False,
            max_iter=2,
        )

        task = Task(
            description="Write one sentence about the sun.",
            expected_output="A single sentence about the sun.",
            agent=agent,
        )

        result = agent.execute_task(task=task)
        assert result is not None

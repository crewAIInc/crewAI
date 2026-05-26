"""Tests for Crew.prompt_file propagation to all components.

Verifies fix for https://github.com/crewAIInc/crewAI/issues/5931
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.utilities.i18n import (
    I18N,
    I18N_DEFAULT,
    get_crew_i18n,
    get_i18n,
    reset_crew_i18n,
    set_crew_i18n,
)

CUSTOM_PROMPTS = os.path.join(os.path.dirname(__file__), "custom_prompts.json")


class TestCrewI18nContextVar:
    """Test the context variable mechanism for crew-scoped I18N."""

    def test_get_crew_i18n_returns_default_when_no_override(self):
        assert get_crew_i18n() is I18N_DEFAULT

    def test_set_and_get_crew_i18n(self):
        custom = get_i18n(CUSTOM_PROMPTS)
        token = set_crew_i18n(custom)
        try:
            assert get_crew_i18n() is custom
            assert get_crew_i18n().slice("role_playing") == "CUSTOM_role_playing"
        finally:
            reset_crew_i18n(token)

    def test_reset_crew_i18n_restores_default(self):
        custom = get_i18n(CUSTOM_PROMPTS)
        token = set_crew_i18n(custom)
        reset_crew_i18n(token)
        assert get_crew_i18n() is I18N_DEFAULT

    def test_nested_set_and_reset(self):
        custom1 = I18N(prompt_file=CUSTOM_PROMPTS)
        custom2 = I18N(prompt_file=CUSTOM_PROMPTS)

        token1 = set_crew_i18n(custom1)
        assert get_crew_i18n() is custom1

        token2 = set_crew_i18n(custom2)
        assert get_crew_i18n() is custom2

        reset_crew_i18n(token2)
        assert get_crew_i18n() is custom1

        reset_crew_i18n(token1)
        assert get_crew_i18n() is I18N_DEFAULT


class TestPromptsUsesCrewI18n:
    """Test that the Prompts class picks up the crew-scoped I18N."""

    def test_prompts_build_prompt_uses_crew_i18n(self):
        from crewai.utilities.prompts import Prompts

        agent = MagicMock()
        agent.role = "TestRole"
        agent.goal = "TestGoal"
        agent.backstory = "TestBackstory"
        agent.skills = []

        custom = get_i18n(CUSTOM_PROMPTS)
        token = set_crew_i18n(custom)
        try:
            prompts = Prompts(agent=agent, has_tools=False)
            result = prompts.task_execution()
            assert "CUSTOM_role_playing" in result.prompt
            assert "CUSTOM_no_tools" in result.prompt
        finally:
            reset_crew_i18n(token)

    def test_prompts_uses_default_when_no_override(self):
        from crewai.utilities.prompts import Prompts

        agent = MagicMock()
        agent.role = "TestRole"
        agent.goal = "TestGoal"
        agent.backstory = "TestBackstory"
        agent.skills = []

        prompts = Prompts(agent=agent, has_tools=False)
        result = prompts.task_execution()
        # Should use default en.json prompts, not custom ones
        assert "CUSTOM_" not in result.prompt
        assert "TestRole" in result.prompt


class TestTaskPromptUsesCrewI18n:
    """Test that Task.prompt() picks up the crew-scoped I18N."""

    def test_task_prompt_uses_crew_i18n(self):
        from crewai.task import Task

        task = Task(
            description="Test task",
            expected_output="Test output",
        )

        custom = get_i18n(CUSTOM_PROMPTS)
        token = set_crew_i18n(custom)
        try:
            prompt = task.prompt()
            assert "CUSTOM_expected_output" in prompt
        finally:
            reset_crew_i18n(token)

    def test_task_prompt_uses_default_when_no_override(self):
        from crewai.task import Task

        task = Task(
            description="Test task",
            expected_output="Test output",
        )
        prompt = task.prompt()
        assert "CUSTOM_" not in prompt
        assert "Test output" in prompt


class TestAgentToolsUseCrewI18n:
    """Test that delegation tools pick up the crew-scoped I18N."""

    def test_agent_tools_use_crew_i18n(self):
        from crewai.agent.core import Agent
        from crewai.tools.agent_tools.agent_tools import AgentTools

        agent = Agent(
            role="TestAgent",
            goal="Test goal",
            backstory="Test backstory",
            llm="gpt-4o",
        )

        custom = get_i18n(CUSTOM_PROMPTS)
        token = set_crew_i18n(custom)
        try:
            tools = AgentTools(agents=[agent]).tools()
            descriptions = [t.description for t in tools]
            assert any("CUSTOM_delegate_work" in d for d in descriptions)
            assert any("CUSTOM_ask_question" in d for d in descriptions)
        finally:
            reset_crew_i18n(token)


class TestCrewKickoffSetsI18nContext:
    """Test that Crew.kickoff() properly sets and resets the I18N context."""

    @patch("crewai.crew.Crew._run_sequential_process")
    @patch("crewai.crews.utils.setup_agents")
    @patch("crewai.crew.Crew.calculate_usage_metrics")
    def test_kickoff_sets_i18n_for_custom_prompt_file(
        self, mock_metrics, mock_setup, mock_seq
    ):
        from crewai.agent.core import Agent
        from crewai.crew import Crew
        from crewai.crews.crew_output import CrewOutput
        from crewai.task import Task

        mock_metrics.return_value = MagicMock()

        from crewai.types.usage_metrics import UsageMetrics

        captured_i18n = []

        def capture_i18n(*args, **kwargs):
            captured_i18n.append(get_crew_i18n())
            return CrewOutput(
                raw="done",
                tasks_output=[],
                json_dict=None,
                pydantic=None,
                token_usage=UsageMetrics(),
            )

        mock_seq.side_effect = capture_i18n

        agent = Agent(
            role="Researcher",
            goal="Research stuff",
            backstory="Expert researcher",
            llm="gpt-4o",
        )
        task = Task(
            description="Do research",
            expected_output="A report",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            prompt_file=CUSTOM_PROMPTS,
        )

        crew.kickoff()

        assert len(captured_i18n) == 1
        assert captured_i18n[0].slice("role_playing") == "CUSTOM_role_playing"
        # After kickoff, context should be reset
        assert get_crew_i18n() is I18N_DEFAULT

    @patch("crewai.crew.Crew._run_sequential_process")
    @patch("crewai.crews.utils.setup_agents")
    @patch("crewai.crew.Crew.calculate_usage_metrics")
    def test_kickoff_resets_i18n_on_exception(
        self, mock_metrics, mock_setup, mock_seq
    ):
        from crewai.agent.core import Agent
        from crewai.crew import Crew
        from crewai.task import Task

        mock_seq.side_effect = RuntimeError("boom")

        agent = Agent(
            role="Researcher",
            goal="Research stuff",
            backstory="Expert researcher",
            llm="gpt-4o",
        )
        task = Task(
            description="Do research",
            expected_output="A report",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            prompt_file=CUSTOM_PROMPTS,
        )

        with pytest.raises(RuntimeError, match="boom"):
            crew.kickoff()

        # Context must be reset even on exception
        assert get_crew_i18n() is I18N_DEFAULT

    @patch("crewai.crew.Crew._run_sequential_process")
    @patch("crewai.crews.utils.setup_agents")
    @patch("crewai.crew.Crew.calculate_usage_metrics")
    def test_kickoff_without_prompt_file_uses_default(
        self, mock_metrics, mock_setup, mock_seq
    ):
        from crewai.agent.core import Agent
        from crewai.crew import Crew
        from crewai.crews.crew_output import CrewOutput
        from crewai.task import Task
        from crewai.types.usage_metrics import UsageMetrics

        mock_metrics.return_value = MagicMock()

        captured_i18n = []

        def capture_i18n(*args, **kwargs):
            captured_i18n.append(get_crew_i18n())
            return CrewOutput(
                raw="done",
                tasks_output=[],
                json_dict=None,
                pydantic=None,
                token_usage=UsageMetrics(),
            )

        mock_seq.side_effect = capture_i18n

        agent = Agent(
            role="Researcher",
            goal="Research stuff",
            backstory="Expert researcher",
            llm="gpt-4o",
        )
        task = Task(
            description="Do research",
            expected_output="A report",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
        )

        crew.kickoff()

        assert len(captured_i18n) == 1
        assert captured_i18n[0] is I18N_DEFAULT

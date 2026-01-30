"""Tests for prompt generation to prevent thought leakage.

These tests verify that:
1. Agents without tools don't get ReAct format instructions
2. The generated prompts don't encourage "Thought:" prefixes that leak into output
3. Real LLM calls produce clean output without internal reasoning
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai import Agent, Crew, Task
from crewai.llm import LLM
from crewai.utilities.prompts import Prompts


class TestNoToolsPromptGeneration:
    """Tests for prompt generation when agent has no tools."""

    def test_no_tools_uses_task_no_tools_slice(self) -> None:
        """Test that agents without tools use task_no_tools slice instead of task."""
        mock_agent = MagicMock()
        mock_agent.role = "Test Agent"
        mock_agent.goal = "Test goal"
        mock_agent.backstory = "Test backstory"

        prompts = Prompts(
            has_tools=False,
            use_native_tool_calling=False,
            use_system_prompt=True,
            agent=mock_agent,
        )

        result = prompts.task_execution()

        # Verify it's a SystemPromptResult with system and user keys
        assert "system" in result
        assert "user" in result
        assert "prompt" in result

        # The user prompt should NOT contain "Thought:" (ReAct format)
        assert "Thought:" not in result["user"]

        # The user prompt should NOT mention tools
        assert "use the tools available" not in result["user"]
        assert "tools available" not in result["user"].lower()

        # The system prompt should NOT contain ReAct format instructions
        assert "Thought:" not in result["system"]
        assert "Final Answer:" not in result["system"]

    def test_no_tools_prompt_is_simple(self) -> None:
        """Test that no-tools prompt is simple and direct."""
        mock_agent = MagicMock()
        mock_agent.role = "Language Detector"
        mock_agent.goal = "Detect language"
        mock_agent.backstory = "Expert linguist"

        prompts = Prompts(
            has_tools=False,
            use_native_tool_calling=False,
            use_system_prompt=True,
            agent=mock_agent,
        )

        result = prompts.task_execution()

        # Should contain the role playing info
        assert "Language Detector" in result["system"]

        # User prompt should be simple with just the task
        assert "Current Task:" in result["user"]
        assert "Provide your complete response:" in result["user"]

    def test_with_tools_uses_task_slice_with_react(self) -> None:
        """Test that agents WITH tools use the task slice (ReAct format)."""
        mock_agent = MagicMock()
        mock_agent.role = "Test Agent"
        mock_agent.goal = "Test goal"
        mock_agent.backstory = "Test backstory"

        prompts = Prompts(
            has_tools=True,
            use_native_tool_calling=False,
            use_system_prompt=True,
            agent=mock_agent,
        )

        result = prompts.task_execution()

        # With tools and ReAct, the prompt SHOULD contain Thought:
        assert "Thought:" in result["user"]

    def test_native_tools_uses_native_task_slice(self) -> None:
        """Test that native tool calling uses native_task slice."""
        mock_agent = MagicMock()
        mock_agent.role = "Test Agent"
        mock_agent.goal = "Test goal"
        mock_agent.backstory = "Test backstory"

        prompts = Prompts(
            has_tools=True,
            use_native_tool_calling=True,
            use_system_prompt=True,
            agent=mock_agent,
        )

        result = prompts.task_execution()

        # Native tool calling should NOT have Thought: in user prompt
        assert "Thought:" not in result["user"]

        # Should NOT have emotional manipulation
        assert "your job depends on it" not in result["user"]


class TestNoThoughtLeakagePatterns:
    """Tests to verify prompts don't encourage thought leakage."""

    def test_no_job_depends_on_it_in_no_tools(self) -> None:
        """Test that 'your job depends on it' is not in no-tools prompts."""
        mock_agent = MagicMock()
        mock_agent.role = "Test"
        mock_agent.goal = "Test"
        mock_agent.backstory = "Test"

        prompts = Prompts(
            has_tools=False,
            use_native_tool_calling=False,
            use_system_prompt=True,
            agent=mock_agent,
        )

        result = prompts.task_execution()

        full_prompt = result["prompt"]
        assert "your job depends on it" not in full_prompt.lower()
        assert "i must use these formats" not in full_prompt.lower()

    def test_no_job_depends_on_it_in_native_task(self) -> None:
        """Test that 'your job depends on it' is not in native task prompts."""
        mock_agent = MagicMock()
        mock_agent.role = "Test"
        mock_agent.goal = "Test"
        mock_agent.backstory = "Test"

        prompts = Prompts(
            has_tools=True,
            use_native_tool_calling=True,
            use_system_prompt=True,
            agent=mock_agent,
        )

        result = prompts.task_execution()

        full_prompt = result["prompt"]
        assert "your job depends on it" not in full_prompt.lower()


class TestRealLLMNoThoughtLeakage:
    """Integration tests with real LLM calls to verify no thought leakage."""

    @pytest.mark.vcr()
    def test_agent_without_tools_no_thought_in_output(self) -> None:
        """Test that agent without tools produces clean output without 'Thought:' prefix."""
        agent = Agent(
            role="Language Detector",
            goal="Detect the language of text",
            backstory="You are an expert linguist who can identify languages.",
            tools=[],  # No tools
            llm=LLM(model="gpt-4o-mini"),
            verbose=False,
        )

        task = Task(
            description="What language is this text written in: 'Hello, how are you?'",
            expected_output="The detected language (e.g., English, Spanish, etc.)",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        assert result.raw is not None

        # The output should NOT start with "Thought:" or contain ReAct artifacts
        output = str(result.raw)
        assert not output.strip().startswith("Thought:")
        assert "Final Answer:" not in output
        assert "I now can give a great answer" not in output

        # Should contain an actual answer about the language
        assert any(
            lang in output.lower()
            for lang in ["english", "en", "language"]
        )

    @pytest.mark.vcr()
    def test_simple_task_clean_output(self) -> None:
        """Test that a simple task produces clean output without internal reasoning."""
        agent = Agent(
            role="Classifier",
            goal="Classify text sentiment",
            backstory="You classify text sentiment accurately.",
            tools=[],
            llm=LLM(model="gpt-4o-mini"),
            verbose=False,
        )

        task = Task(
            description="Classify the sentiment of: 'I love this product!'",
            expected_output="One word: positive, negative, or neutral",
            agent=agent,
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        assert result is not None
        output = str(result.raw).strip().lower()

        # Output should be clean - just the classification
        assert not output.startswith("thought:")
        assert "final answer:" not in output

        # Should contain the actual classification
        assert any(
            sentiment in output
            for sentiment in ["positive", "negative", "neutral"]
        )

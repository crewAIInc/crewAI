"""Integration tests for structured context compaction (summarize_messages).

These tests use VCR cassettes to record and replay real API calls.

To record cassettes for the first time:
    PYTEST_VCR_RECORD_MODE=all pytest lib/crewai/tests/utilities/test_summarize_integration.py -v

To replay from cassettes:
    pytest lib/crewai/tests/utilities/test_summarize_integration.py -v
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.llm import LLM
from crewai.task import Task
from crewai.utilities.agent_utils import summarize_messages
from crewai.utilities.i18n import I18N


def _build_conversation_messages(
    *, include_system: bool = True, include_files: bool = False
) -> list[dict[str, Any]]:
    """Build a realistic multi-turn conversation for summarization tests."""
    messages: list[dict[str, Any]] = []

    if include_system:
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a research assistant specializing in AI topics. "
                    "Your goal is to find accurate, up-to-date information."
                ),
            }
        )

    user_msg: dict[str, Any] = {
        "role": "user",
        "content": (
            "Research the latest developments in large language models. "
            "Focus on architecture improvements and training techniques."
        ),
    }
    if include_files:
        user_msg["files"] = {"reference.pdf": MagicMock()}
    messages.append(user_msg)

    messages.append(
        {
            "role": "assistant",
            "content": (
                "I'll research the latest developments in large language models. "
                "Based on my knowledge, recent advances include:\n"
                "1. Mixture of Experts (MoE) architectures\n"
                "2. Improved attention mechanisms like Flash Attention\n"
                "3. Better training data curation techniques\n"
                "4. Constitutional AI and RLHF improvements"
            ),
        }
    )

    messages.append(
        {
            "role": "user",
            "content": "Can you go deeper on the MoE architectures? What are the key papers?",
        }
    )

    messages.append(
        {
            "role": "assistant",
            "content": (
                "Key papers on Mixture of Experts:\n"
                "- Switch Transformers (Google, 2021) - simplified MoE routing\n"
                "- GShard - scaling to 600B parameters\n"
                "- Mixtral (Mistral AI) - open-source MoE model\n"
                "The main advantage is computational efficiency: "
                "only a subset of experts is activated per token."
            ),
        }
    )

    return messages


class TestSummarizeDirectOpenAI:
    """Test direct summarize_messages calls with OpenAI."""

    @pytest.mark.vcr()
    def test_summarize_direct_openai(self) -> None:
        """Test summarize_messages with gpt-4o-mini preserves system messages."""
        llm = LLM(model="gpt-4o-mini", temperature=0)
        i18n = I18N()
        messages = _build_conversation_messages(include_system=True)

        original_system_content = messages[0]["content"]

        summarize_messages(
            messages=messages,
            llm=llm,
            callbacks=[],
            i18n=i18n,
        )

        # System message should be preserved
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == original_system_content

        # Summary should be a user message
        summary_msg = messages[-1]
        assert summary_msg["role"] == "user"
        assert len(summary_msg["content"]) > 0


class TestSummarizeDirectAnthropic:
    """Test direct summarize_messages calls with Anthropic."""

    @pytest.mark.vcr()
    def test_summarize_direct_anthropic(self) -> None:
        """Test summarize_messages with claude-3-5-haiku."""
        llm = LLM(model="anthropic/claude-3-5-haiku-latest", temperature=0)
        i18n = I18N()
        messages = _build_conversation_messages(include_system=True)

        summarize_messages(
            messages=messages,
            llm=llm,
            callbacks=[],
            i18n=i18n,
        )

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        summary_msg = messages[-1]
        assert summary_msg["role"] == "user"
        assert len(summary_msg["content"]) > 0


class TestSummarizeDirectGemini:
    """Test direct summarize_messages calls with Gemini."""

    @pytest.mark.vcr()
    def test_summarize_direct_gemini(self) -> None:
        """Test summarize_messages with gemini-2.0-flash."""
        llm = LLM(model="gemini/gemini-2.0-flash", temperature=0)
        i18n = I18N()
        messages = _build_conversation_messages(include_system=True)

        summarize_messages(
            messages=messages,
            llm=llm,
            callbacks=[],
            i18n=i18n,
        )

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        summary_msg = messages[-1]
        assert summary_msg["role"] == "user"
        assert len(summary_msg["content"]) > 0


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

        # This may or may not trigger compaction depending on actual response sizes.
        # The test verifies the code path doesn't crash.
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


class TestSummarizePreservesFiles:
    """Test that files are preserved through real summarization."""

    @pytest.mark.vcr()
    def test_summarize_preserves_files_integration(self) -> None:
        """Test that file references survive a real summarization call."""
        llm = LLM(model="gpt-4o-mini", temperature=0)
        i18n = I18N()
        messages = _build_conversation_messages(
            include_system=True, include_files=True
        )

        summarize_messages(
            messages=messages,
            llm=llm,
            callbacks=[],
            i18n=i18n,
        )

        # System message preserved
        assert messages[0]["role"] == "system"

        # Files should be on the summary message
        summary_msg = messages[-1]
        assert "files" in summary_msg
        assert "reference.pdf" in summary_msg["files"]

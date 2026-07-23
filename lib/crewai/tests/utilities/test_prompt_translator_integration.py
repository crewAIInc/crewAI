"""Integration tests for Agent with auto_translate_prompt feature."""

from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent
from crewai.utilities.prompt_translator import clear_translation_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear translation cache before each test."""
    clear_translation_cache()
    yield
    clear_translation_cache()


class TestAgentAutoTranslatePrompt:
    """Tests for the auto_translate_prompt parameter on Agent."""

    def test_default_enabled(self):
        """auto_translate_prompt should default to True."""
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
        )
        assert agent.auto_translate_prompt is True

    def test_disable(self):
        """auto_translate_prompt can be disabled."""
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            auto_translate_prompt=False,
        )
        assert agent.auto_translate_prompt is False

    def test_glossary_default_none(self):
        """prompt_glossary should default to None."""
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
        )
        assert agent.prompt_glossary is None

    def test_glossary_set(self):
        """prompt_glossary can be set."""
        glossary = {"API Key": "API Key", "crewAI": "crewAI"}
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            prompt_glossary=glossary,
        )
        assert agent.prompt_glossary == glossary


class TestAgentPromptTranslation:
    """Tests for prompt translation integration in Agent._build_execution_prompt."""

    @patch("crewai.utilities.prompt_translator.optimize_system_prompt")
    def test_translation_called_when_enabled(self, mock_optimize):
        """optimize_system_prompt should be called when auto_translate_prompt=True."""
        mock_optimize.side_effect = lambda prompt, model, glossary=None: prompt

        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            auto_translate_prompt=True,
        )

        # Access the private method to test prompt building
        from crewai.tools.structured_tool import CrewStructuredTool

        prompt, _, _ = agent._build_execution_prompt([])

        # optimize_system_prompt should have been called
        assert mock_optimize.called

    @patch("crewai.utilities.prompt_translator.optimize_system_prompt")
    def test_translation_not_called_when_disabled(self, mock_optimize):
        """optimize_system_prompt should NOT be called when auto_translate_prompt=False."""
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            auto_translate_prompt=False,
        )

        prompt, _, _ = agent._build_execution_prompt([])

        mock_optimize.assert_not_called()

    @patch("crewai.utilities.prompt_translator.optimize_system_prompt")
    def test_glossary_passed_to_optimizer(self, mock_optimize):
        """prompt_glossary should be passed to optimize_system_prompt."""
        mock_optimize.side_effect = lambda prompt, model, glossary=None: prompt

        glossary = {"API Key": "API Key"}
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            auto_translate_prompt=True,
            prompt_glossary=glossary,
        )

        prompt, _, _ = agent._build_execution_prompt([])

        # Check that optimize_system_prompt was called with the glossary
        for call in mock_optimize.call_args_list:
            assert call[1].get("glossary") == glossary or (
                len(call[0]) >= 3 and call[0][2] == glossary
            )

    @patch("crewai.utilities.prompt_translator.optimize_system_prompt")
    def test_model_name_extracted_from_llm(self, mock_optimize):
        """The model name should be extracted from the LLM instance."""
        mock_optimize.side_effect = lambda prompt, model, glossary=None: prompt

        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            auto_translate_prompt=True,
        )

        prompt, _, _ = agent._build_execution_prompt([])

        # The model name should have been passed to optimize_system_prompt
        for call in mock_optimize.call_args_list:
            model_arg = call[0][1] if len(call[0]) > 1 else call[1].get("model_name")
            assert "gpt-4o" in str(model_arg)

    @patch("crewai.utilities.prompt_translator.optimize_system_prompt")
    def test_system_prompt_result_optimized(self, mock_optimize):
        """When SystemPromptResult is returned, system prompt should be optimized."""
        mock_optimize.side_effect = lambda prompt, model, glossary=None, llm_caller=None: f"optimized:{prompt}"

        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm="gpt-4o",
            auto_translate_prompt=True,
            use_system_prompt=True,
        )

        prompt, _, _ = agent._build_execution_prompt([])

        # The system prompt should carry the "optimized:" prefix
        from crewai.utilities.prompts import SystemPromptResult

        assert isinstance(prompt, SystemPromptResult)
        assert prompt.system.startswith("optimized:"), (
            f"Expected system prompt to start with 'optimized:', got: {prompt.system!r}"
        )

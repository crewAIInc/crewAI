"""Tests for stop sequences synchronization across LLM providers.

This test module verifies that the stop sequences are properly synchronized
between the `stop` attribute (set by CrewAgentExecutor) and the provider-specific
`stop_sequences` attribute (sent to the API) for Anthropic, Bedrock, and Gemini providers.

Issue: https://github.com/crewAIInc/crewAI/issues/3836
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestAnthropicStopSequencesSync:
    """Test stop sequences synchronization for AnthropicCompletion."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client to avoid network calls."""
        with patch("crewai.llms.providers.anthropic.completion.Anthropic") as mock:
            yield mock

    def test_stop_property_getter_returns_stop_sequences(self, mock_anthropic_client):
        """Test that getting stop returns stop_sequences."""
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicCompletion(
                model="claude-3-5-sonnet-20241022",
                stop_sequences=[r"\nObservation:"],
            )

            assert llm.stop == [r"\nObservation:"]
            assert llm.stop_sequences == [r"\nObservation:"]
            assert llm.stop is llm.stop_sequences

    def test_stop_property_setter_syncs_with_stop_sequences(self, mock_anthropic_client):
        """Test that setting stop updates stop_sequences."""
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022")

            llm.stop = [r"\nObservation:", r"\nFinal Answer:"]
            assert llm.stop_sequences == [r"\nObservation:", r"\nFinal Answer:"]
            assert llm.stop == llm.stop_sequences

    def test_stop_property_setter_handles_string(self, mock_anthropic_client):
        """Test that setting stop with a string converts to list."""
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022")

            llm.stop = r"\nObservation:"
            assert llm.stop_sequences == [r"\nObservation:"]
            assert llm.stop == [r"\nObservation:"]

    def test_stop_property_setter_handles_none(self, mock_anthropic_client):
        """Test that setting stop to None clears stop_sequences."""
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicCompletion(
                model="claude-3-5-sonnet-20241022",
                stop_sequences=[r"\nObservation:"],
            )

            llm.stop = None
            assert llm.stop_sequences == []
            assert llm.stop == []

    def test_crew_agent_executor_pattern(self, mock_anthropic_client):
        """Test the pattern used by CrewAgentExecutor to set stop sequences."""
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022")

            existing_stop = getattr(llm, "stop", [])
            new_stops = [r"\nObservation:"]
            llm.stop = list(
                set(existing_stop + new_stops if isinstance(existing_stop, list) else new_stops)
            )

            assert llm.stop_sequences == [r"\nObservation:"]
            assert llm.stop == llm.stop_sequences

    def test_prepare_completion_params_includes_stop_sequences(
        self, mock_anthropic_client
    ):
        """Test that _prepare_completion_params includes stop_sequences in params."""
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            llm = AnthropicCompletion(model="claude-3-5-sonnet-20241022")

            llm.stop = [r"\nObservation:"]

            params = llm._prepare_completion_params(
                messages=[{"role": "user", "content": "test"}]
            )

            assert "stop_sequences" in params
            assert params["stop_sequences"] == [r"\nObservation:"]


class TestBedrockStopSequencesSync:
    """Test stop sequences synchronization for BedrockCompletion."""

    @pytest.fixture
    def mock_bedrock_session(self):
        """Mock boto3 Session to avoid AWS calls."""
        with patch("crewai.llms.providers.bedrock.completion.Session") as mock:
            mock_client = MagicMock()
            mock.return_value.client.return_value = mock_client
            yield mock

    def test_stop_property_getter_returns_stop_sequences(self, mock_bedrock_session):
        """Test that getting stop returns stop_sequences."""
        from crewai.llms.providers.bedrock.completion import BedrockCompletion

        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            stop_sequences=[r"\nObservation:"],
        )

        assert llm.stop == [r"\nObservation:"]
        assert llm.stop_sequences == [r"\nObservation:"]
        assert llm.stop is llm.stop_sequences

    def test_stop_property_setter_syncs_with_stop_sequences(self, mock_bedrock_session):
        """Test that setting stop updates stop_sequences."""
        from crewai.llms.providers.bedrock.completion import BedrockCompletion

        llm = BedrockCompletion(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

        llm.stop = [r"\nObservation:", r"\nFinal Answer:"]
        assert llm.stop_sequences == [r"\nObservation:", r"\nFinal Answer:"]
        assert llm.stop == llm.stop_sequences

    def test_stop_property_setter_handles_string(self, mock_bedrock_session):
        """Test that setting stop with a string converts to list."""
        from crewai.llms.providers.bedrock.completion import BedrockCompletion

        llm = BedrockCompletion(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

        llm.stop = r"\nObservation:"
        assert llm.stop_sequences == [r"\nObservation:"]
        assert llm.stop == [r"\nObservation:"]

    def test_stop_property_setter_handles_none(self, mock_bedrock_session):
        """Test that setting stop to None clears stop_sequences."""
        from crewai.llms.providers.bedrock.completion import BedrockCompletion

        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            stop_sequences=[r"\nObservation:"],
        )

        llm.stop = None
        assert llm.stop_sequences == []
        assert llm.stop == []

    def test_crew_agent_executor_pattern(self, mock_bedrock_session):
        """Test the pattern used by CrewAgentExecutor to set stop sequences."""
        from crewai.llms.providers.bedrock.completion import BedrockCompletion

        llm = BedrockCompletion(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

        existing_stop = getattr(llm, "stop", [])
        new_stops = [r"\nObservation:"]
        llm.stop = list(
            set(existing_stop + new_stops if isinstance(existing_stop, list) else new_stops)
        )

        assert llm.stop_sequences == [r"\nObservation:"]
        assert llm.stop == llm.stop_sequences

    def test_get_inference_config_includes_stop_sequences(self, mock_bedrock_session):
        """Test that _get_inference_config includes stopSequences."""
        from crewai.llms.providers.bedrock.completion import BedrockCompletion

        llm = BedrockCompletion(model="anthropic.claude-3-5-sonnet-20241022-v2:0")

        llm.stop = [r"\nObservation:"]

        config = llm._get_inference_config()

        assert "stopSequences" in config
        assert config["stopSequences"] == [r"\nObservation:"]


class TestGeminiStopSequencesSync:
    """Test stop sequences synchronization for GeminiCompletion."""

    @pytest.fixture
    def mock_gemini_client(self):
        """Mock Google Gen AI client to avoid network calls."""
        with patch("crewai.llms.providers.gemini.completion.genai") as mock:
            mock_client = MagicMock()
            mock.Client.return_value = mock_client
            yield mock

    def test_stop_property_getter_returns_stop_sequences(self, mock_gemini_client):
        """Test that getting stop returns stop_sequences."""
        from crewai.llms.providers.gemini.completion import GeminiCompletion

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = GeminiCompletion(
                model="gemini-2.0-flash-001",
                stop_sequences=[r"\nObservation:"],
            )

            assert llm.stop == [r"\nObservation:"]
            assert llm.stop_sequences == [r"\nObservation:"]
            assert llm.stop is llm.stop_sequences

    def test_stop_property_setter_syncs_with_stop_sequences(self, mock_gemini_client):
        """Test that setting stop updates stop_sequences."""
        from crewai.llms.providers.gemini.completion import GeminiCompletion

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = GeminiCompletion(model="gemini-2.0-flash-001")

            llm.stop = [r"\nObservation:", r"\nFinal Answer:"]
            assert llm.stop_sequences == [r"\nObservation:", r"\nFinal Answer:"]
            assert llm.stop == llm.stop_sequences

    def test_stop_property_setter_handles_string(self, mock_gemini_client):
        """Test that setting stop with a string converts to list."""
        from crewai.llms.providers.gemini.completion import GeminiCompletion

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = GeminiCompletion(model="gemini-2.0-flash-001")

            llm.stop = r"\nObservation:"
            assert llm.stop_sequences == [r"\nObservation:"]
            assert llm.stop == [r"\nObservation:"]

    def test_stop_property_setter_handles_none(self, mock_gemini_client):
        """Test that setting stop to None clears stop_sequences."""
        from crewai.llms.providers.gemini.completion import GeminiCompletion

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = GeminiCompletion(
                model="gemini-2.0-flash-001",
                stop_sequences=[r"\nObservation:"],
            )

            llm.stop = None
            assert llm.stop_sequences == []
            assert llm.stop == []

    def test_crew_agent_executor_pattern(self, mock_gemini_client):
        """Test the pattern used by CrewAgentExecutor to set stop sequences."""
        from crewai.llms.providers.gemini.completion import GeminiCompletion

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = GeminiCompletion(model="gemini-2.0-flash-001")

            existing_stop = getattr(llm, "stop", [])
            new_stops = [r"\nObservation:"]
            llm.stop = list(
                set(existing_stop + new_stops if isinstance(existing_stop, list) else new_stops)
            )

            assert llm.stop_sequences == [r"\nObservation:"]
            assert llm.stop == llm.stop_sequences

    def test_prepare_generation_config_includes_stop_sequences(self, mock_gemini_client):
        """Test that _prepare_generation_config includes stop_sequences."""
        from crewai.llms.providers.gemini.completion import GeminiCompletion

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            llm = GeminiCompletion(model="gemini-2.0-flash-001")

            llm.stop = [r"\nObservation:"]

            config = llm._prepare_generation_config()

            assert "stop_sequences" in config
            assert config["stop_sequences"] == [r"\nObservation:"]

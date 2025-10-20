"""Tests for Google Generative AI embedder configuration (issue #3741)."""

from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Crew, Task


class TestGoogleGenerativeAIEmbedder:
    """Test Google Generative AI embedder configuration formats."""

    @patch("crewai.crew.Knowledge")
    @patch("crewai.crew.ShortTermMemory")
    @patch("crewai.crew.LongTermMemory")
    @patch("crewai.crew.EntityMemory")
    def test_crew_with_google_generativeai_flat_config(
        self, mock_entity_memory, mock_long_term_memory, mock_short_term_memory, mock_knowledge
    ):
        """Test that Crew accepts google-generativeai embedder with flat config format (issue #3741)."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        embedder_config = {
            "provider": "google-generativeai",
            "api_key": "test-gemini-key",
            "model_name": "models/text-embedding-004",
        }

        crew = Crew(
            agents=[agent],
            tasks=[task],
            embedder=embedder_config,
        )

        expected_normalized_config = {
            "provider": "google-generativeai",
            "config": {
                "api_key": "test-gemini-key",
                "model_name": "models/text-embedding-004",
            },
        }
        assert crew.embedder == expected_normalized_config

    @patch("crewai.crew.Knowledge")
    @patch("crewai.crew.ShortTermMemory")
    @patch("crewai.crew.LongTermMemory")
    @patch("crewai.crew.EntityMemory")
    def test_crew_with_google_generativeai_nested_config(
        self, mock_entity_memory, mock_long_term_memory, mock_short_term_memory, mock_knowledge
    ):
        """Test that Crew accepts google-generativeai embedder with nested config format."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
        )

        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )

        embedder_config = {
            "provider": "google-generativeai",
            "config": {
                "api_key": "test-gemini-key",
                "model_name": "models/text-embedding-004",
            },
        }

        crew = Crew(
            agents=[agent],
            tasks=[task],
            embedder=embedder_config,
        )

        assert crew.embedder == embedder_config

    def test_generativeai_provider_spec_validation(self):
        """Test that GenerativeAiProviderSpec validates correctly with optional config."""
        from crewai.rag.embeddings.types import GenerativeAiProviderSpec

        flat_spec: GenerativeAiProviderSpec = {
            "provider": "google-generativeai",
        }
        assert flat_spec["provider"] == "google-generativeai"

        nested_spec: GenerativeAiProviderSpec = {
            "provider": "google-generativeai",
            "config": {
                "api_key": "test-key",
                "model_name": "models/text-embedding-004",
            },
        }
        assert nested_spec["provider"] == "google-generativeai"
        assert nested_spec["config"]["api_key"] == "test-key"

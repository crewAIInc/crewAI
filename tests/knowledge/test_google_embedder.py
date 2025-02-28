from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
from crewai.utilities.constants import GOOGLE_EMBEDDER_PACKAGE_ERROR_MSG


def test_google_embedder_missing_package():
    """Test that a helpful error message is displayed when google-generativeai is not installed."""
    # Create a simple agent and task
    agent = Agent(
        role="Test Agent",
        goal="Test the knowledge component",
        backstory="I am a test agent",
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    # Create a mock JSONKnowledgeSource
    json_knowledge_source = MagicMock(spec=JSONKnowledgeSource)

    # Mock the GoogleGenerativeAiEmbeddingFunction to raise the import error
    with patch("chromadb.utils.embedding_functions.google_embedding_function.GoogleGenerativeAiEmbeddingFunction.__init__") as mock_init:
        mock_init.side_effect = ValueError("The Google Generative AI python package is not installed. Please install it with `pip install google-generativeai`")
        
        # Mock the logger to capture the error message
        with patch("crewai.utilities.logger.Logger.log") as mock_log:
            # Create a crew with Google embedder
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                knowledge_sources=[json_knowledge_source],
                embedder={
                    "provider": "google",
                    "config": {
                        "api_key": "fake-api-key",
                        "model": 'models/embedding-001'
                    }
                }
            )
            
            # Verify that the error message was logged correctly
            mock_log.assert_any_call(
                "error",
                GOOGLE_EMBEDDER_PACKAGE_ERROR_MSG,
                color="red"
            )


def test_google_embedder_invalid_api_key():
    """Test that a warning is logged when an invalid API key is provided."""
    # Create a simple agent and task
    agent = Agent(
        role="Test Agent",
        goal="Test the knowledge component",
        backstory="I am a test agent",
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    # Create a mock JSONKnowledgeSource
    json_knowledge_source = MagicMock(spec=JSONKnowledgeSource)

    # Mock the GoogleGenerativeAiEmbeddingFunction to raise an exception for invalid API key
    with patch("chromadb.utils.embedding_functions.google_embedding_function.GoogleGenerativeAiEmbeddingFunction.__init__") as mock_init:
        mock_init.side_effect = ValueError("Invalid API key")
        
        # Mock the logger to capture the warning message
        with patch("crewai.utilities.logger.Logger.log") as mock_log:
            # Create a crew with Google embedder
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                knowledge_sources=[json_knowledge_source],
                embedder={
                    "provider": "google",
                    "config": {
                        "api_key": "invalid-api-key",
                        "model": 'models/embedding-001'
                    }
                }
            )
            
            # Verify that the warning message was logged correctly
            mock_log.assert_any_call(
                "warning",
                "Failed to init knowledge: Invalid API key",
                color="yellow"
            )


def test_google_embedder_invalid_model():
    """Test that a warning is logged when an invalid model is provided."""
    # Create a simple agent and task
    agent = Agent(
        role="Test Agent",
        goal="Test the knowledge component",
        backstory="I am a test agent",
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    # Create a mock JSONKnowledgeSource
    json_knowledge_source = MagicMock(spec=JSONKnowledgeSource)

    # Mock the GoogleGenerativeAiEmbeddingFunction to raise an exception for invalid model
    with patch("chromadb.utils.embedding_functions.google_embedding_function.GoogleGenerativeAiEmbeddingFunction.__init__") as mock_init:
        mock_init.side_effect = ValueError("Invalid model name")
        
        # Mock the logger to capture the warning message
        with patch("crewai.utilities.logger.Logger.log") as mock_log:
            # Create a crew with Google embedder
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                knowledge_sources=[json_knowledge_source],
                embedder={
                    "provider": "google",
                    "config": {
                        "api_key": "fake-api-key",
                        "model": 'invalid-model'
                    }
                }
            )
            
            # Verify that the warning message was logged correctly
            mock_log.assert_any_call(
                "warning",
                "Failed to init knowledge: Invalid model name",
                color="yellow"
            )

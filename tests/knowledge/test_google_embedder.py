import pytest
from unittest.mock import patch, MagicMock
from crewai import Crew, Agent, Task
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource

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
                "Google AI Studio embedder requires the google-generativeai package. "
                "Please install it with `pip install google-generativeai`",
                color="red"
            )

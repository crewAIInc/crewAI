"""Integration tests for Google Vertex embeddings with Crew memory.

These tests make real API calls and use VCR to record/replay responses.
"""

import os
from unittest.mock import patch

import pytest

from crewai import Agent, Crew, Task
from crewai.memory.unified_memory import Memory


@pytest.fixture(autouse=True)
def setup_vertex_ai_env():
    """Set up environment for Vertex AI tests.

    Sets GOOGLE_GENAI_USE_VERTEXAI=true to ensure the SDK uses the Vertex AI
    backend (aiplatform.googleapis.com) which matches the VCR cassettes.
    Also mocks GOOGLE_API_KEY if not already set.
    """
    env_updates = {"GOOGLE_GENAI_USE_VERTEXAI": "true"}

    # Add a mock API key if none exists
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" not in os.environ:
        env_updates["GOOGLE_API_KEY"] = "test-key"

    with patch.dict(os.environ, env_updates):
        yield


@pytest.fixture
def google_vertex_embedder_config():
    """Fixture providing Google Vertex embedder configuration."""
    return {
        "provider": "google-vertex",
        "config": {
            # "api_key": os.getenv("GOOGLE_API_KEY", "test-key"),
            # "project_id": os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0393486657"),
            "model_name": "gemini-embedding-001",
        },
    }


@pytest.fixture
def simple_agent():
    """Fixture providing a simple test agent."""
    return Agent(
        role="Research Assistant",
        goal="Help with research tasks",
        backstory="You are a helpful research assistant.",
        verbose=False,
    )


@pytest.fixture
def simple_task(simple_agent):
    """Fixture providing a simple test task."""
    return Task(
        description="Summarize the key points about artificial intelligence in one sentence.",
        expected_output="A one sentence summary about AI.",
        agent=simple_agent,
    )


@pytest.mark.vcr()
@pytest.mark.timeout(120)
def test_crew_memory_with_google_vertex_embedder(
    google_vertex_embedder_config, simple_agent, simple_task
) -> None:
    """Test that Crew with google-vertex embedder stores memories after task execution."""
    from crewai.rag.embeddings.factory import build_embedder

    embedder = build_embedder(google_vertex_embedder_config)
    memory = Memory(embedder=embedder)

    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        memory=memory,
        verbose=True,
    )

    assert crew._memory is memory

    result = crew.kickoff()

    assert result is not None
    assert result.raw is not None
    assert len(result.raw) > 0

    # Verify memories were actually written to storage
    info = crew._memory.info("/")
    assert info.record_count > 0, (
        f"Expected memories to be saved after crew kickoff, "
        f"but found {info.record_count} records"
    )


@pytest.mark.vcr()
@pytest.mark.timeout(120)
def test_crew_memory_with_google_vertex_project_id(simple_agent, simple_task) -> None:
    """Test Crew memory with Google Vertex using project_id authentication."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        pytest.skip("GOOGLE_CLOUD_PROJECT environment variable not set")

    from crewai.rag.embeddings.factory import build_embedder

    embedder_config = {
        "provider": "google-vertex",
        "config": {
            "project_id": project_id,
            "location": "us-central1",
            "model_name": "gemini-embedding-001",
        },
    }

    embedder = build_embedder(embedder_config)
    memory = Memory(embedder=embedder)

    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        memory=memory,
        verbose=False,
    )

    assert crew._memory is memory

    result = crew.kickoff()

    assert result is not None
    assert result.raw is not None

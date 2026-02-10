"""Integration tests for Google Vertex embeddings with Crew memory.

These tests make real API calls and use VCR to record/replay responses.
The memory save path (extract_memories + remember) requires LLM and embedding
API calls that are difficult to capture in VCR cassettes (GCP metadata auth,
embedding endpoints). We mock those paths and verify the crew pipeline works
end-to-end while testing memory storage separately with a fake embedder.
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

    # Add a mock API key
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
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0393486657"),
            "location": "us-central1",
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


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake embeddings for testing storage without real API calls."""
    return [[0.1] * 1536 for _ in texts]


@pytest.mark.vcr()
@pytest.mark.timeout(120)
def test_crew_memory_with_google_vertex_embedder(
    google_vertex_embedder_config, simple_agent, simple_task
) -> None:
    """Test that Crew with google-vertex embedder runs and that memory storage works.

    The crew kickoff uses VCR-recorded LLM responses. The memory save path
    (extract_memories + remember) is mocked during kickoff because it requires
    embedding/auth API calls not in the cassette. After kickoff we verify
    memory storage works by calling remember() directly with a fake embedder.
    """
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

    # Mock _save_to_memory during kickoff so it doesn't make embedding API calls
    # that VCR can't replay (GCP metadata auth, embedding endpoints).
    with patch(
        "crewai.agents.agent_builder.base_agent_executor_mixin.CrewAgentExecutorMixin._save_to_memory"
    ):
        result = crew.kickoff()

    assert result is not None
    assert result.raw is not None
    assert len(result.raw) > 0

    # Now verify the memory storage path works by calling remember() directly
    # with a fake embedder that doesn't need real API calls.
    memory._embedder = _fake_embedder

    # Also mock the LLM analysis (analyze_for_save) so remember() doesn't need
    # an LLM call -- pass all fields explicitly to skip analysis.
    record = memory.remember(
        content=f"AI summary: {result.raw[:100]}",
        scope="/test",
        categories=["ai", "summary"],
        importance=0.7,
    )
    assert record is not None
    assert record.scope == "/test"

    info = memory.info("/")
    assert info.record_count > 0, (
        f"Expected memories to be saved after manual remember(), "
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

    with patch(
        "crewai.agents.agent_builder.base_agent_executor_mixin.CrewAgentExecutorMixin._save_to_memory"
    ):
        result = crew.kickoff()

    assert result is not None
    assert result.raw is not None

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
            # "api_key": os.getenv("GOOGLE_API_KEY", "test-key"),
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


@pytest.mark.vcr()
@pytest.mark.timeout(120)
def test_crew_memory_with_google_vertex_embedder(
    google_vertex_embedder_config, simple_agent, simple_task
) -> None:
    """Test that Crew with google-vertex embedder stores memories after task execution."""
    from crewai.rag.embeddings.factory import build_embedder

    embedder = build_embedder(google_vertex_embedder_config)
    memory = Memory(embedder=embedder)

    print(f"\n[DEBUG] Memory storage type: {type(memory._storage).__name__}")
    print(f"[DEBUG] Memory storage path: {getattr(memory._storage, '_path', 'N/A')}")
    print(f"[DEBUG] CREWAI_STORAGE_DIR={os.environ.get('CREWAI_STORAGE_DIR', 'not set')}")

    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        memory=memory,
        verbose=True,
    )

    print(f"[DEBUG] crew._memory is memory: {crew._memory is memory}")
    print(f"[DEBUG] crew._memory type: {type(crew._memory).__name__}")
    assert crew._memory is memory

    result = crew.kickoff()

    assert result is not None
    assert result.raw is not None
    assert len(result.raw) > 0
    print(f"[DEBUG] Result raw length: {len(result.raw)}")

    # Verify memories were actually written to storage
    info = crew._memory.info("/")
    print(f"[DEBUG] Memory info after kickoff: record_count={info.record_count}, "
          f"categories={info.categories}, child_scopes={info.child_scopes}")

    # Debug: try extract_memories + remember manually to see if it works
    if info.record_count == 0:
        print("[DEBUG] No records found -- attempting manual extract_memories + remember")
        try:
            extracted = memory.extract_memories(f"Task result: {result.raw}")
            print(f"[DEBUG] extract_memories returned {len(extracted)} items: {extracted[:3]}")
        except Exception as e:
            print(f"[DEBUG] extract_memories FAILED: {type(e).__name__}: {e}")
            extracted = []

        for i, mem in enumerate(extracted):
            try:
                record = memory.remember(mem)
                print(f"[DEBUG] remember({i}) succeeded: id={record.id}, scope={record.scope}")
            except Exception as e:
                print(f"[DEBUG] remember({i}) FAILED: {type(e).__name__}: {e}")

        info_after = memory.info("/")
        print(f"[DEBUG] Memory info after manual save: record_count={info_after.record_count}")

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

"""Integration tests for Google Vertex embeddings with Crew memory.

These tests make real API calls and use VCR to record/replay responses.
"""

import os
import threading
from collections import defaultdict
from unittest.mock import patch

import pytest

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)


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
            "api_key": os.getenv("GOOGLE_API_KEY", "test-key"),
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
@pytest.mark.timeout(120)  # Longer timeout for VCR recording
def test_crew_memory_with_google_vertex_embedder(
    google_vertex_embedder_config, simple_agent, simple_task
) -> None:
    """Test that Crew with memory=True works with google-vertex embedder and memory is used."""
    # Track memory events
    events: dict[str, list] = defaultdict(list)
    condition = threading.Condition()

    @crewai_event_bus.on(MemorySaveStartedEvent)
    def on_save_started(source, event):
        with condition:
            events["MemorySaveStartedEvent"].append(event)
            condition.notify()

    @crewai_event_bus.on(MemorySaveCompletedEvent)
    def on_save_completed(source, event):
        with condition:
            events["MemorySaveCompletedEvent"].append(event)
            condition.notify()

    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        memory=True,
        embedder=google_vertex_embedder_config,
        verbose=False,
    )

    result = crew.kickoff()

    assert result is not None
    assert result.raw is not None
    assert len(result.raw) > 0

    with condition:
        success = condition.wait_for(
            lambda: len(events["MemorySaveCompletedEvent"]) >= 1,
            timeout=10,
        )

    assert success, "Timeout waiting for memory save events - memory may not be working"
    assert len(events["MemorySaveStartedEvent"]) >= 1, "No memory save started events"
    assert len(events["MemorySaveCompletedEvent"]) >= 1, "Memory save completed events"


@pytest.mark.vcr()
@pytest.mark.timeout(120)
def test_crew_memory_with_google_vertex_project_id(simple_agent, simple_task) -> None:
    """Test Crew memory with Google Vertex using project_id authentication."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        pytest.skip("GOOGLE_CLOUD_PROJECT environment variable not set")

    # Track memory events
    events: dict[str, list] = defaultdict(list)
    condition = threading.Condition()

    @crewai_event_bus.on(MemorySaveStartedEvent)
    def on_save_started(source, event):
        with condition:
            events["MemorySaveStartedEvent"].append(event)
            condition.notify()

    @crewai_event_bus.on(MemorySaveCompletedEvent)
    def on_save_completed(source, event):
        with condition:
            events["MemorySaveCompletedEvent"].append(event)
            condition.notify()

    embedder_config = {
        "provider": "google-vertex",
        "config": {
            "project_id": project_id,
            "location": "us-central1",
            "model_name": "gemini-embedding-001",
        },
    }

    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        memory=True,
        embedder=embedder_config,
        verbose=False,
    )

    result = crew.kickoff()

    # Verify basic result
    assert result is not None
    assert result.raw is not None

    # Wait for memory save events
    with condition:
        success = condition.wait_for(
            lambda: len(events["MemorySaveCompletedEvent"]) >= 1,
            timeout=10,
        )

    # Verify memory was actually used
    assert success, "Timeout waiting for memory save events - memory may not be working"
    assert len(events["MemorySaveStartedEvent"]) >= 1, "No memory save started events"
    assert len(events["MemorySaveCompletedEvent"]) >= 1, "No memory save completed events"

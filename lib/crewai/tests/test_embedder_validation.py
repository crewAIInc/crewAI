"""Tests for embedder configuration validation (Issue #3755)."""

import pytest
from pydantic_core import ValidationError

from crewai import Agent, Crew, Task


@pytest.fixture
def simple_agent():
    """Create a simple agent for testing."""
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory"
    )


@pytest.fixture
def simple_task(simple_agent):
    """Create a simple task for testing."""
    return Task(
        description="Test task",
        expected_output="Test output",
        agent=simple_agent
    )


def test_invalid_embedder_provider_name(simple_agent, simple_task):
    """Test that an invalid provider name gives a clear error message."""
    invalid_config = {
        "provider": "invalid-provider",
        "config": {
            "api_key": "test_key",
            "model_name": "test_model"
        }
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    assert "Invalid embedder provider: 'invalid-provider'" in error_message
    assert "Valid providers are:" in error_message
    assert "google-generativeai" in error_message
    assert "openai" in error_message


def test_google_generativeai_missing_config_field(simple_agent, simple_task):
    """Test that missing config field for google-generativeai gives a clear error."""
    invalid_config = {
        "provider": "google-generativeai",
        "model_name": "models/text-embedding-004",
        "api_key": "test_key"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    assert "Invalid embedder configuration for provider 'google-generativeai'" in error_message
    assert "missing the required 'config' field" in error_message
    assert "Expected structure:" in error_message
    assert '"provider": "google-generativeai"' in error_message
    assert '"config"' in error_message


def test_google_vertex_missing_config_field(simple_agent, simple_task):
    """Test that missing config field for google-vertex gives a clear error."""
    invalid_config = {
        "provider": "google-vertex",
        "model_name": "textembedding-gecko",
        "api_key": "test_key"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    assert "Invalid embedder configuration for provider 'google-vertex'" in error_message
    assert "missing the required 'config' field" in error_message
    assert "Expected structure:" in error_message


def test_valid_google_generativeai_config(simple_agent, simple_task):
    """Test that a valid google-generativeai config is accepted."""
    valid_config = {
        "provider": "google-generativeai",
        "config": {
            "api_key": "test_key",
            "model_name": "models/embedding-001"
        }
    }
    
    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        embedder=valid_config
    )
    
    assert crew.embedder == valid_config


def test_valid_openai_config(simple_agent, simple_task):
    """Test that a valid openai config is accepted."""
    valid_config = {
        "provider": "openai",
        "api_key": "test_key",
        "model": "text-embedding-3-small"
    }
    
    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        embedder=valid_config
    )
    
    assert crew.embedder is not None
    assert crew.embedder["provider"] == "openai"


def test_valid_ollama_config(simple_agent, simple_task):
    """Test that a valid ollama config is accepted."""
    valid_config = {
        "provider": "ollama",
        "model": "nomic-embed-text"
    }
    
    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        embedder=valid_config
    )
    
    assert crew.embedder is not None
    assert crew.embedder["provider"] == "ollama"


def test_none_embedder_config(simple_agent, simple_task):
    """Test that None embedder config is accepted."""
    crew = Crew(
        agents=[simple_agent],
        tasks=[simple_task],
        embedder=None
    )
    
    assert crew.embedder is None


def test_embedder_config_without_provider_field(simple_agent, simple_task):
    """Test that config without provider field is handled by Pydantic."""
    invalid_config = {
        "api_key": "test_key",
        "model_name": "test_model"
    }
    
    with pytest.raises(ValidationError):
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )


def test_sentence_transformer_missing_config_field(simple_agent, simple_task):
    """Test that missing config field for sentence-transformer gives a clear error."""
    invalid_config = {
        "provider": "sentence-transformer",
        "model_name": "all-MiniLM-L6-v2"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    assert "Invalid embedder configuration for provider 'sentence-transformer'" in error_message
    assert "missing the required 'config' field" in error_message


def test_voyageai_missing_config_field(simple_agent, simple_task):
    """Test that missing config field for voyageai gives a clear error."""
    invalid_config = {
        "provider": "voyageai",
        "api_key": "test_key"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    assert "Invalid embedder configuration for provider 'voyageai'" in error_message
    assert "missing the required 'config' field" in error_message


def test_error_shows_received_config(simple_agent, simple_task):
    """Test that error message shows the received configuration."""
    invalid_config = {
        "provider": "google-generativeai",
        "model_name": "models/text-embedding-004",
        "api_key": "test_key"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    assert "But received:" in error_message
    assert '"model_name": "models/text-embedding-004"' in error_message


def test_single_validation_error_not_multiple(simple_agent, simple_task):
    """Test that we get a single clear error, not multiple confusing errors."""
    invalid_config = {
        "provider": "google-generativeai",
        "model_name": "models/text-embedding-004",
        "api_key": "test_key"
    }
    
    with pytest.raises(ValidationError) as exc_info:
        Crew(
            agents=[simple_agent],
            tasks=[simple_task],
            embedder=invalid_config
        )
    
    error_message = str(exc_info.value)
    error_count = error_message.count("validation error")
    
    assert error_count == 1, f"Expected 1 validation error, got {error_count}"
    
    assert "AzureProviderSpec" not in error_message
    assert "BedrockProviderSpec" not in error_message
    assert "CohereProviderSpec" not in error_message

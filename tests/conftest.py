# conftest.py
import os
import pytest
from dotenv import load_dotenv

load_result = load_dotenv(override=True)

@pytest.fixture(autouse=True)
def setup_test_env():
    """Configure test environment to use Ollama as the default embedding provider."""
    # Store original environment variables
    original_env = {
        "CREWAI_EMBEDDING_PROVIDER": os.environ.get("CREWAI_EMBEDDING_PROVIDER"),
        "CREWAI_EMBEDDING_MODEL": os.environ.get("CREWAI_EMBEDDING_MODEL"),
        "CREWAI_OLLAMA_URL": os.environ.get("CREWAI_OLLAMA_URL"),
    }
    
    # Set test environment
    os.environ["CREWAI_EMBEDDING_PROVIDER"] = "ollama"
    os.environ["CREWAI_EMBEDDING_MODEL"] = "llama2"
    os.environ["CREWAI_OLLAMA_URL"] = "http://localhost:11434/api/embeddings"
    
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

from crewai.utilities.embedding_configurator import EmbeddingConfigurator
import pytest
from unittest.mock import patch

def test_ollama_embedder_url_config():
    configurator = EmbeddingConfigurator()
    
    test_cases = [
        # Test default URL
        {
            "config": {"provider": "ollama", "config": {"model": "test-model"}},
            "expected_url": "http://localhost:11434/api/embeddings"
        },
        # Test legacy url key
        {
            "config": {"provider": "ollama", "config": {"model": "test-model", "url": "http://custom:11434"}},
            "expected_url": "http://custom:11434"
        },
        # Test api_url key
        {
            "config": {"provider": "ollama", "config": {"model": "test-model", "api_url": "http://api:11434"}},
            "expected_url": "http://api:11434"
        },
        # Test base_url key
        {
            "config": {"provider": "ollama", "config": {"model": "test-model", "base_url": "http://base:11434"}},
            "expected_url": "http://base:11434"
        },
        # Test api_base key
        {
            "config": {"provider": "ollama", "config": {"model": "test-model", "api_base": "http://base-api:11434"}},
            "expected_url": "http://base-api:11434"
        },
        # Test URL precedence order
        {
            "config": {
                "provider": "ollama",
                "config": {
                    "model": "test-model",
                    "url": "http://url:11434",
                    "api_url": "http://api:11434",
                    "base_url": "http://base:11434",
                    "api_base": "http://base-api:11434"
                }
            },
            "expected_url": "http://url:11434"  # url key should have highest precedence
        }
    ]

    for test_case in test_cases:
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            configurator.configure_embedder(test_case["config"])
            mock_ollama.assert_called_once()
            _, kwargs = mock_ollama.call_args
            assert kwargs["url"] == test_case["expected_url"]
            mock_ollama.reset_mock()

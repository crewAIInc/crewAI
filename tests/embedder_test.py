from unittest.mock import patch

import pytest

from crewai.utilities.embedding_configurator import EmbeddingConfigurator


@pytest.mark.parametrize(
    "test_case",
    [
        pytest.param(
            {
                "config": {"provider": "ollama", "config": {"model": "test-model"}},
                "expected_url": "http://localhost:11434/api/embeddings"
            },
            id="default_url"
        ),
        pytest.param(
            {
                "config": {"provider": "ollama", "config": {"model": "test-model", "url": "http://custom:11434"}},
                "expected_url": "http://custom:11434"
            },
            id="legacy_url"
        ),
        pytest.param(
            {
                "config": {"provider": "ollama", "config": {"model": "test-model", "api_url": "http://api:11434"}},
                "expected_url": "http://api:11434"
            },
            id="api_url"
        ),
        pytest.param(
            {
                "config": {"provider": "ollama", "config": {"model": "test-model", "base_url": "http://base:11434"}},
                "expected_url": "http://base:11434"
            },
            id="base_url"
        ),
        pytest.param(
            {
                "config": {"provider": "ollama", "config": {"model": "test-model", "api_base": "http://base-api:11434"}},
                "expected_url": "http://base-api:11434"
            },
            id="api_base"
        ),
        pytest.param(
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
                "expected_url": "http://url:11434"
            },
            id="url_precedence"
        ),
    ]
)
def test_ollama_embedder_url_config(test_case):
    configurator = EmbeddingConfigurator()
    with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
        configurator.configure_embedder(test_case["config"])
        mock_ollama.assert_called_once()
        _, kwargs = mock_ollama.call_args
        assert kwargs["url"] == test_case["expected_url"]
        mock_ollama.reset_mock()

def test_ollama_embedder_invalid_url():
    configurator = EmbeddingConfigurator()
    with pytest.raises(ValueError, match="Invalid URL format"):
        configurator.configure_embedder({
            "provider": "ollama",
            "config": {
                "model": "test-model",
                "url": "invalid-url"
            }
        })

def test_ollama_embedder_missing_model():
    configurator = EmbeddingConfigurator()
    with pytest.raises(ValueError, match="Model name is required"):
        configurator.configure_embedder({
            "provider": "ollama",
            "config": {
                "url": "http://valid:11434"
            }
        })

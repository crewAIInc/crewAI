import os
import pytest
from unittest.mock import patch
from crewai.utilities.embedding_configurator import EmbeddingConfigurator


@pytest.mark.url_configuration
class TestOllamaEmbeddingConfigurator:
    def setup_method(self):
        self.configurator = EmbeddingConfigurator()

    @patch.dict(os.environ, {}, clear=True)
    def test_ollama_default_url(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://localhost:11434/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://custom-ollama:8080/api/embeddings"}, clear=True)
    def test_ollama_respects_api_base_env_var(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://custom-ollama:8080/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://env-ollama:8080"}, clear=True)
    def test_ollama_config_url_overrides_env_var(self):
        config = {
            "provider": "ollama", 
            "config": {
                "model": "llama2",
                "url": "http://config-ollama:9090/api/embeddings"
            }
        }
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://config-ollama:9090/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://env-ollama:8080/api/embeddings"}, clear=True)
    def test_ollama_config_api_base_overrides_env_var(self):
        config = {
            "provider": "ollama", 
            "config": {
                "model": "llama2",
                "api_base": "http://config-ollama:9090/api/embeddings"
            }
        }
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://config-ollama:9090/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_ollama_url_priority_order(self):
        config = {
            "provider": "ollama", 
            "config": {
                "model": "llama2",
                "url": "http://url-config:1111/api/embeddings",
                "api_base": "http://api-base-config:2222/api/embeddings"
            }
        }
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://url-config:1111/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://localhost:11434/api/embeddings"}, clear=True)
    def test_ollama_handles_full_url_in_api_base(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://localhost:11434/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://localhost:11434/api/embeddings"}, clear=True)
    def test_ollama_uses_provided_url_as_is(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://localhost:11434/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_ollama_config_api_base_without_url(self):
        config = {
            "provider": "ollama", 
            "config": {
                "model": "llama2",
                "api_base": "http://config-ollama:9090/api/embeddings"
            }
        }
        
        with patch("chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://config-ollama:9090/api/embeddings",
                model_name="llama2"
            )

@pytest.mark.error_handling
class TestOllamaErrorHandling:
    def setup_method(self):
        self.configurator = EmbeddingConfigurator()

    @pytest.mark.parametrize("invalid_url", [
        "not-a-url",
        "ftp://invalid-scheme",
        "http://",
        "://missing-scheme",
        "http:///missing-netloc",
    ])
    def test_invalid_url_raises_error(self, invalid_url):
        """Test that invalid URLs raise ValueError with clear error message."""
        config = {
            "provider": "ollama",
            "config": {
                "model": "llama2",
                "url": invalid_url
            }
        }

        with pytest.raises(ValueError, match="Invalid Ollama API URL"):
            self.configurator.configure_embedder(config)

    @pytest.mark.parametrize("invalid_api_base", [
        "not-a-url",
        "ftp://invalid-scheme", 
        "http://",
        "://missing-scheme",
    ])
    def test_invalid_api_base_raises_error(self, invalid_api_base):
        """Test that invalid api_base URLs raise ValueError with clear error message."""
        config = {
            "provider": "ollama",
            "config": {
                "model": "llama2",
                "api_base": invalid_api_base
            }
        }

        with pytest.raises(ValueError, match="Invalid Ollama API URL"):
            self.configurator.configure_embedder(config)

    @patch.dict(os.environ, {"API_BASE": "not-a-valid-url"}, clear=True)
    def test_invalid_env_var_raises_error(self):
        """Test that invalid API_BASE environment variable raises ValueError."""
        config = {"provider": "ollama", "config": {"model": "llama2"}}

        with pytest.raises(ValueError, match="Invalid Ollama API URL"):
            self.configurator.configure_embedder(config)

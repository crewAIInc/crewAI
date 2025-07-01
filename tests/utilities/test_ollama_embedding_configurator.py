import os
import pytest
from unittest.mock import patch, MagicMock
from crewai.utilities.embedding_configurator import EmbeddingConfigurator


class TestOllamaEmbeddingConfigurator:
    def setup_method(self):
        self.configurator = EmbeddingConfigurator()

    @patch.dict(os.environ, {}, clear=True)
    def test_ollama_default_url(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://localhost:11434/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://custom-ollama:8080"}, clear=True)
    def test_ollama_respects_api_base_env_var(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
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
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://config-ollama:9090/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://env-ollama:8080"}, clear=True)
    def test_ollama_config_api_base_overrides_env_var(self):
        config = {
            "provider": "ollama", 
            "config": {
                "model": "llama2",
                "api_base": "http://config-ollama:9090"
            }
        }
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
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
                "api_base": "http://api-base-config:2222"
            }
        }
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://url-config:1111/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://localhost:11434/"}, clear=True)
    def test_ollama_handles_trailing_slash_in_api_base(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://localhost:11434/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://localhost:11434/api/embeddings"}, clear=True)
    def test_ollama_handles_full_url_in_api_base(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://localhost:11434/api/embeddings",
                model_name="llama2"
            )

    @patch.dict(os.environ, {"API_BASE": "http://localhost:11434"}, clear=True)
    def test_ollama_api_base_without_trailing_slash(self):
        config = {"provider": "ollama", "config": {"model": "llama2"}}
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
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
                "api_base": "http://config-ollama:9090"
            }
        }
        
        with patch("crewai.utilities.embedding_configurator.OllamaEmbeddingFunction") as mock_ollama:
            self.configurator.configure_embedder(config)
            mock_ollama.assert_called_once_with(
                url="http://config-ollama:9090/api/embeddings",
                model_name="llama2"
            )

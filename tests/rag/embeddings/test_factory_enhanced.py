"""Enhanced tests for embedding function factory."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.factory import (  # type: ignore[import-untyped]
    get_embedding_function,
)
from crewai.rag.embeddings.types import EmbeddingOptions  # type: ignore[import-untyped]


def test_get_embedding_function_default() -> None:
    """Test default embedding function when no config provided."""
    with patch("crewai.rag.embeddings.factory.OpenAIEmbeddingFunction") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        with patch(
            "crewai.rag.embeddings.factory.os.getenv", return_value="test-api-key"
        ):
            result = get_embedding_function()

        mock_openai.assert_called_once_with(
            api_key="test-api-key", model_name="text-embedding-3-small"
        )
        assert result == mock_instance


def test_get_embedding_function_with_embedding_options() -> None:
    """Test embedding function creation with EmbeddingOptions object."""
    with patch("crewai.rag.embeddings.factory.OpenAIEmbeddingFunction") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        options = EmbeddingOptions(
            provider="openai", api_key="test-key", model="text-embedding-3-large"
        )

        result = get_embedding_function(options)

        call_kwargs = mock_openai.call_args.kwargs
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"].get_secret_value() == "test-key"
        # OpenAI uses model_name parameter, not model
        assert result == mock_instance


def test_get_embedding_function_sentence_transformer() -> None:
    """Test sentence transformer embedding function."""
    with patch(
        "crewai.rag.embeddings.factory.SentenceTransformerEmbeddingFunction"
    ) as mock_st:
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance

        config = {"provider": "sentence-transformer", "model_name": "all-MiniLM-L6-v2"}

        result = get_embedding_function(config)

        mock_st.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        assert result == mock_instance


def test_get_embedding_function_ollama() -> None:
    """Test Ollama embedding function."""
    with patch("crewai.rag.embeddings.factory.OllamaEmbeddingFunction") as mock_ollama:
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance

        config = {
            "provider": "ollama",
            "model_name": "nomic-embed-text",
            "url": "http://localhost:11434",
        }

        result = get_embedding_function(config)

        mock_ollama.assert_called_once_with(
            model_name="nomic-embed-text", url="http://localhost:11434"
        )
        assert result == mock_instance


def test_get_embedding_function_cohere() -> None:
    """Test Cohere embedding function."""
    with patch("crewai.rag.embeddings.factory.CohereEmbeddingFunction") as mock_cohere:
        mock_instance = MagicMock()
        mock_cohere.return_value = mock_instance

        config = {
            "provider": "cohere",
            "api_key": "cohere-key",
            "model_name": "embed-english-v3.0",
        }

        result = get_embedding_function(config)

        mock_cohere.assert_called_once_with(
            api_key="cohere-key", model_name="embed-english-v3.0"
        )
        assert result == mock_instance


def test_get_embedding_function_huggingface() -> None:
    """Test HuggingFace embedding function."""
    with patch("crewai.rag.embeddings.factory.HuggingFaceEmbeddingFunction") as mock_hf:
        mock_instance = MagicMock()
        mock_hf.return_value = mock_instance

        config = {
            "provider": "huggingface",
            "api_key": "hf-token",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }

        result = get_embedding_function(config)

        mock_hf.assert_called_once_with(
            api_key="hf-token", model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        assert result == mock_instance


def test_get_embedding_function_onnx() -> None:
    """Test ONNX embedding function."""
    with patch("crewai.rag.embeddings.factory.ONNXMiniLM_L6_V2") as mock_onnx:
        mock_instance = MagicMock()
        mock_onnx.return_value = mock_instance

        config = {"provider": "onnx"}

        result = get_embedding_function(config)

        mock_onnx.assert_called_once()
        assert result == mock_instance


def test_get_embedding_function_google_palm() -> None:
    """Test Google PaLM embedding function."""
    with patch(
        "crewai.rag.embeddings.factory.GooglePalmEmbeddingFunction"
    ) as mock_palm:
        mock_instance = MagicMock()
        mock_palm.return_value = mock_instance

        config = {"provider": "google-palm", "api_key": "palm-key"}

        result = get_embedding_function(config)

        mock_palm.assert_called_once_with(api_key="palm-key")
        assert result == mock_instance


def test_get_embedding_function_amazon_bedrock() -> None:
    """Test Amazon Bedrock embedding function."""
    with patch(
        "crewai.rag.embeddings.factory.AmazonBedrockEmbeddingFunction"
    ) as mock_bedrock:
        mock_instance = MagicMock()
        mock_bedrock.return_value = mock_instance

        config = {
            "provider": "amazon-bedrock",
            "region_name": "us-west-2",
            "model_name": "amazon.titan-embed-text-v1",
        }

        result = get_embedding_function(config)

        mock_bedrock.assert_called_once_with(
            region_name="us-west-2", model_name="amazon.titan-embed-text-v1"
        )
        assert result == mock_instance


def test_get_embedding_function_jina() -> None:
    """Test Jina embedding function."""
    with patch("crewai.rag.embeddings.factory.JinaEmbeddingFunction") as mock_jina:
        mock_instance = MagicMock()
        mock_jina.return_value = mock_instance

        config = {
            "provider": "jina",
            "api_key": "jina-key",
            "model_name": "jina-embeddings-v2-base-en",
        }

        result = get_embedding_function(config)

        mock_jina.assert_called_once_with(
            api_key="jina-key", model_name="jina-embeddings-v2-base-en"
        )
        assert result == mock_instance


def test_get_embedding_function_unsupported_provider() -> None:
    """Test handling of unsupported provider."""
    config = {"provider": "unsupported-provider"}

    with pytest.raises(ValueError, match="Unsupported provider: unsupported-provider"):
        get_embedding_function(config)


def test_get_embedding_function_config_modification() -> None:
    """Test that original config dict is not modified."""
    original_config = {
        "provider": "openai",
        "api_key": "test-key",
        "model": "text-embedding-3-small",
    }
    config_copy = original_config.copy()

    with patch("crewai.rag.embeddings.factory.OpenAIEmbeddingFunction"):
        get_embedding_function(config_copy)

    assert config_copy == original_config


def test_get_embedding_function_exclude_none_values() -> None:
    """Test that None values are excluded from embedding function calls."""
    with patch("crewai.rag.embeddings.factory.OpenAIEmbeddingFunction") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance

        options = EmbeddingOptions(provider="openai", api_key="test-key", model=None)

        result = get_embedding_function(options)

        call_kwargs = mock_openai.call_args.kwargs
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"].get_secret_value() == "test-key"
        assert "model" not in call_kwargs
        assert result == mock_instance


def test_get_embedding_function_instructor() -> None:
    """Test Instructor embedding function."""
    with patch(
        "crewai.rag.embeddings.factory.InstructorEmbeddingFunction"
    ) as mock_instructor:
        mock_instance = MagicMock()
        mock_instructor.return_value = mock_instance

        config = {"provider": "instructor", "model_name": "hkunlp/instructor-large"}

        result = get_embedding_function(config)

        mock_instructor.assert_called_once_with(model_name="hkunlp/instructor-large")
        assert result == mock_instance


def test_get_embedding_function_watson() -> None:
    """Test Watson embedding function."""
    with patch("crewai.rag.embeddings.factory._create_watson_embedding_function") as mock_watson:
        mock_instance = MagicMock()
        mock_watson.return_value = mock_instance

        config = {
            "provider": "watson",
            "api_key": "watson-api-key",
            "api_url": "https://watson-url.com",
            "project_id": "watson-project-id",
            "model_name": "ibm/slate-125m-english-rtrvr",
        }

        result = get_embedding_function(config)

        mock_watson.assert_called_once_with(
            api_key="watson-api-key",
            api_url="https://watson-url.com",
            project_id="watson-project-id",
            model_name="ibm/slate-125m-english-rtrvr",
        )
        assert result == mock_instance


def test_get_embedding_function_watson_missing_dependencies() -> None:
    """Test Watson embedding function with missing dependencies."""
    with patch("crewai.rag.embeddings.factory._create_watson_embedding_function") as mock_watson:
        mock_watson.side_effect = ImportError(
            "IBM Watson dependencies are not installed. Please install them to use Watson embedding."
        )

        config = {
            "provider": "watson",
            "api_key": "watson-api-key",
            "api_url": "https://watson-url.com",
            "project_id": "watson-project-id",
            "model_name": "ibm/slate-125m-english-rtrvr",
        }

        with pytest.raises(ImportError, match="IBM Watson dependencies are not installed"):
            get_embedding_function(config)


def test_get_embedding_function_watson_with_embedding_options() -> None:
    """Test Watson embedding function with EmbeddingOptions object."""
    with patch("crewai.rag.embeddings.factory._create_watson_embedding_function") as mock_watson:
        mock_instance = MagicMock()
        mock_watson.return_value = mock_instance

        options = EmbeddingOptions(
            provider="watson",
            api_key="watson-key",
            model_name="ibm/slate-125m-english-rtrvr"
        )

        result = get_embedding_function(options)

        call_kwargs = mock_watson.call_args.kwargs
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"].get_secret_value() == "watson-key"
        assert call_kwargs["model_name"] == "ibm/slate-125m-english-rtrvr"
        assert result == mock_instance

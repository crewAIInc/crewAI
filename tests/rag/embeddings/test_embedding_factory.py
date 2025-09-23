"""Enhanced tests for embedding function factory."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

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
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openai
        mock_providers.__contains__.return_value = True

        options = EmbeddingOptions(
            provider="openai",
            api_key=SecretStr("test-key"),
            model_name="text-embedding-3-large",
        )

        result = get_embedding_function(options)

        call_kwargs = mock_openai.call_args.kwargs
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"].get_secret_value() == "test-key"
        assert "model_name" in call_kwargs
        assert call_kwargs["model_name"] == "text-embedding-3-large"
        assert result == mock_instance


def test_get_embedding_function_sentence_transformer() -> None:
    """Test sentence transformer embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_st = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_st
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "sentence-transformer",
            "config": {"model_name": "all-MiniLM-L6-v2"},
        }

        result = get_embedding_function(config)

        mock_st.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        assert result == mock_instance


def test_get_embedding_function_ollama() -> None:
    """Test Ollama embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_ollama = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_ollama
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "ollama",
            "config": {
                "model_name": "nomic-embed-text",
                "url": "http://localhost:11434",
            },
        }

        result = get_embedding_function(config)

        mock_ollama.assert_called_once_with(
            model_name="nomic-embed-text", url="http://localhost:11434"
        )
        assert result == mock_instance


def test_get_embedding_function_cohere() -> None:
    """Test Cohere embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_cohere = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_cohere
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "cohere",
            "config": {"api_key": "cohere-key", "model_name": "embed-english-v3.0"},
        }

        result = get_embedding_function(config)

        mock_cohere.assert_called_once_with(
            api_key="cohere-key", model_name="embed-english-v3.0"
        )
        assert result == mock_instance


def test_get_embedding_function_huggingface() -> None:
    """Test HuggingFace embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_hf = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_hf
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "huggingface",
            "config": {
                "api_key": "hf-token",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }

        result = get_embedding_function(config)

        mock_hf.assert_called_once_with(
            api_key="hf-token", model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        assert result == mock_instance


def test_get_embedding_function_onnx() -> None:
    """Test ONNX embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_onnx = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_onnx
        mock_providers.__contains__.return_value = True

        config = {"provider": "onnx"}

        result = get_embedding_function(config)

        mock_onnx.assert_called_once()
        assert result == mock_instance


def test_get_embedding_function_google_palm() -> None:
    """Test Google PaLM embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_palm = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_palm
        mock_providers.__contains__.return_value = True

        config = {"provider": "google-palm", "config": {"api_key": "palm-key"}}

        result = get_embedding_function(config)

        mock_palm.assert_called_once_with(api_key="palm-key")
        assert result == mock_instance


def test_get_embedding_function_amazon_bedrock() -> None:
    """Test Amazon Bedrock embedding function with explicit session."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_bedrock = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_bedrock
        mock_providers.__contains__.return_value = True

        # Provide an explicit session to avoid boto3 import
        mock_session = MagicMock()
        config = {
            "provider": "amazon-bedrock",
            "config": {
                "session": mock_session,
                "region_name": "us-west-2",
                "model_name": "amazon.titan-embed-text-v1",
            },
        }

        result = get_embedding_function(config)

        mock_bedrock.assert_called_once_with(
            session=mock_session,
            region_name="us-west-2",
            model_name="amazon.titan-embed-text-v1",
        )
        assert result == mock_instance


def test_get_embedding_function_jina() -> None:
    """Test Jina embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_jina = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_jina
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "jina",
            "config": {
                "api_key": "jina-key",
                "model_name": "jina-embeddings-v2-base-en",
            },
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
        "config": {"api_key": "test-key", "model": "text-embedding-3-small"},
    }
    config_copy = original_config.copy()

    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openai
        mock_providers.__contains__.return_value = True

        get_embedding_function(config_copy)

    assert config_copy == original_config


def test_get_embedding_function_exclude_none_values() -> None:
    """Test that None values are excluded from embedding function calls."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openai
        mock_providers.__contains__.return_value = True

        options = EmbeddingOptions(
            provider="openai", api_key=SecretStr("test-key"), model_name=None
        )

        result = get_embedding_function(options)

        call_kwargs = mock_openai.call_args.kwargs
        assert "api_key" in call_kwargs
        assert call_kwargs["api_key"].get_secret_value() == "test-key"
        assert "model_name" not in call_kwargs
        assert result == mock_instance


def test_get_embedding_function_instructor() -> None:
    """Test Instructor embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_instructor = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_instructor
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "instructor",
            "config": {"model_name": "hkunlp/instructor-large"},
        }

        result = get_embedding_function(config)

        mock_instructor.assert_called_once_with(model_name="hkunlp/instructor-large")
        assert result == mock_instance


def test_get_embedding_function_google_generativeai() -> None:
    """Test Google Generative AI embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_google = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_google
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "google-generativeai",
            "config": {"api_key": "google-key", "model_name": "models/embedding-001"},
        }

        result = get_embedding_function(config)

        mock_google.assert_called_once_with(
            api_key="google-key", model_name="models/embedding-001"
        )
        assert result == mock_instance


def test_get_embedding_function_google_vertex() -> None:
    """Test Google Vertex AI embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_vertex = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_vertex
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "google-vertex",
            "config": {
                "api_key": "vertex-key",
                "project_id": "my-project",
                "region": "us-central1",
            },
        }

        result = get_embedding_function(config)

        mock_vertex.assert_called_once_with(
            api_key="vertex-key", project_id="my-project", region="us-central1"
        )
        assert result == mock_instance


def test_get_embedding_function_roboflow() -> None:
    """Test Roboflow embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_roboflow = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_roboflow
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "roboflow",
            "config": {
                "api_key": "roboflow-key",
                "api_url": "https://infer.roboflow.com",
            },
        }

        result = get_embedding_function(config)

        mock_roboflow.assert_called_once_with(
            api_key="roboflow-key", api_url="https://infer.roboflow.com"
        )
        assert result == mock_instance


def test_get_embedding_function_openclip() -> None:
    """Test OpenCLIP embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openclip = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openclip
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "openclip",
            "config": {"model_name": "ViT-B-32", "checkpoint": "laion2b_s34b_b79k"},
        }

        result = get_embedding_function(config)

        mock_openclip.assert_called_once_with(
            model_name="ViT-B-32", checkpoint="laion2b_s34b_b79k"
        )
        assert result == mock_instance


def test_get_embedding_function_text2vec() -> None:
    """Test Text2Vec embedding function."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_text2vec = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_text2vec
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "text2vec",
            "config": {"model_name": "shibing624/text2vec-base-chinese"},
        }

        result = get_embedding_function(config)

        mock_text2vec.assert_called_once_with(
            model_name="shibing624/text2vec-base-chinese"
        )
        assert result == mock_instance


def test_model_to_model_name_conversion() -> None:
    """Test that 'model' field is converted to 'model_name' for nested config."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openai
        mock_providers.__contains__.return_value = True

        config = {
            "provider": "openai",
            "config": {"api_key": "test-key", "model": "text-embedding-3-small"},
        }

        result = get_embedding_function(config)

        mock_openai.assert_called_once_with(
            api_key="test-key", model_name="text-embedding-3-small"
        )
        assert result == mock_instance


def test_api_key_injection_from_env_openai() -> None:
    """Test that OpenAI API key is injected from environment when not provided."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openai
        mock_providers.__contains__.return_value = True

        with patch("crewai.rag.embeddings.factory.os.getenv") as mock_getenv:
            mock_getenv.return_value = "env-openai-key"

            config = {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            }

            result = get_embedding_function(config)

            mock_getenv.assert_called_with("OPENAI_API_KEY")
            mock_openai.assert_called_once_with(
                api_key="env-openai-key", model_name="text-embedding-3-small"
            )
            assert result == mock_instance


def test_api_key_injection_from_env_cohere() -> None:
    """Test that Cohere API key is injected from environment when not provided."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_cohere = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_cohere
        mock_providers.__contains__.return_value = True

        with patch("crewai.rag.embeddings.factory.os.getenv") as mock_getenv:
            mock_getenv.return_value = "env-cohere-key"

            config = {
                "provider": "cohere",
                "config": {"model_name": "embed-english-v3.0"},
            }

            result = get_embedding_function(config)

            mock_getenv.assert_called_with("COHERE_API_KEY")
            mock_cohere.assert_called_once_with(
                api_key="env-cohere-key", model_name="embed-english-v3.0"
            )
            assert result == mock_instance


def test_api_key_not_injected_when_provided() -> None:
    """Test that API key from config takes precedence over environment."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_openai = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_openai
        mock_providers.__contains__.return_value = True

        with patch("crewai.rag.embeddings.factory.os.getenv") as mock_getenv:
            mock_getenv.return_value = "env-key"

            config = {
                "provider": "openai",
                "config": {"api_key": "config-key", "model": "text-embedding-3-small"},
            }

            result = get_embedding_function(config)

            mock_openai.assert_called_once_with(
                api_key="config-key", model_name="text-embedding-3-small"
            )
            assert result == mock_instance


def test_amazon_bedrock_session_injection() -> None:
    """Test that boto3 session is automatically created for amazon-bedrock."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_bedrock = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_bedrock
        mock_providers.__contains__.return_value = True

        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            mock_session = MagicMock()
            mock_boto3.Session.return_value = mock_session

            config = {
                "provider": "amazon-bedrock",
                "config": {"model_name": "amazon.titan-embed-text-v1"},
            }

            result = get_embedding_function(config)

            mock_boto3.Session.assert_called_once()
            mock_bedrock.assert_called_once_with(
                session=mock_session, model_name="amazon.titan-embed-text-v1"
            )
            assert result == mock_instance


def test_amazon_bedrock_session_not_injected_when_provided() -> None:
    """Test that provided session is used for amazon-bedrock."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_instance = MagicMock()
        mock_bedrock = MagicMock(return_value=mock_instance)
        mock_providers.__getitem__.return_value = mock_bedrock
        mock_providers.__contains__.return_value = True

        existing_session = MagicMock()
        config = {
            "provider": "amazon-bedrock",
            "config": {
                "session": existing_session,
                "model_name": "amazon.titan-embed-text-v1",
            },
        }

        result = get_embedding_function(config)

        mock_bedrock.assert_called_once_with(
            session=existing_session, model_name="amazon.titan-embed-text-v1"
        )
        assert result == mock_instance


def test_amazon_bedrock_boto3_import_error() -> None:
    """Test error handling when boto3 is not installed."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_providers.__contains__.return_value = True

        with patch.dict("sys.modules", {"boto3": None}):
            config = {
                "provider": "amazon-bedrock",
                "config": {"model_name": "amazon.titan-embed-text-v1"},
            }

            with pytest.raises(
                ImportError, match="boto3 is required for amazon-bedrock"
            ):
                get_embedding_function(config)


def test_amazon_bedrock_session_creation_error() -> None:
    """Test error handling when AWS session creation fails."""
    with patch("crewai.rag.embeddings.factory.EMBEDDING_PROVIDERS") as mock_providers:
        mock_providers.__contains__.return_value = True

        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            mock_boto3.Session.side_effect = Exception("AWS credentials not configured")

            config = {
                "provider": "amazon-bedrock",
                "config": {"model_name": "amazon.titan-embed-text-v1"},
            }

            with pytest.raises(ValueError, match="Failed to create AWS session"):
                get_embedding_function(config)


def test_invalid_config_format() -> None:
    """Test error handling for invalid config format."""
    config = {
        "provider": "openai",
        "api_key": "test-key",
        "model": "text-embedding-3-small",
    }

    with pytest.raises(ValueError, match="Invalid embedder configuration format"):
        get_embedding_function(config)

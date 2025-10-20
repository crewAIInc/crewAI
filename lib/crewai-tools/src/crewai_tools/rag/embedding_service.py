"""
Enhanced embedding service that leverages CrewAI's existing embedding providers.
This replaces the litellm-based EmbeddingService with a more flexible architecture.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers."""

    provider: str = Field(description="Embedding provider name")
    model: str = Field(description="Model name to use")
    api_key: str | None = Field(default=None, description="API key for the provider")
    timeout: float | None = Field(
        default=30.0, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, description="Maximum number of retries")
    batch_size: int = Field(
        default=100, description="Batch size for processing multiple texts"
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific configuration"
    )


class EmbeddingService:
    """
    Enhanced embedding service that uses CrewAI's existing embedding providers.

    Supports multiple providers:
    - openai: OpenAI embeddings (text-embedding-3-small, text-embedding-3-large, etc.)
    - voyageai: Voyage AI embeddings (voyage-2, voyage-large-2, etc.)
    - cohere: Cohere embeddings (embed-english-v3.0, embed-multilingual-v3.0, etc.)
    - google-generativeai: Google Gemini embeddings (models/embedding-001, etc.)
    - google-vertex: Google Vertex embeddings (models/embedding-001, etc.)
    - huggingface: Hugging Face embeddings (sentence-transformers/all-MiniLM-L6-v2, etc.)
    - jina: Jina embeddings (jina-embeddings-v2-base-en, etc.)
    - ollama: Ollama embeddings (nomic-embed-text, etc.)
    - openai: OpenAI embeddings (text-embedding-3-small, text-embedding-3-large, etc.)
    - roboflow: Roboflow embeddings (roboflow-embeddings-v2-base-en, etc.)
    - voyageai: Voyage AI embeddings (voyage-2, voyage-large-2, etc.)
    - watsonx: Watson X embeddings (ibm/slate-125m-english-rtrvr, etc.)
    - custom: Custom embeddings (embedding_callable, etc.)
    - sentence-transformer: Sentence Transformers embeddings (all-MiniLM-L6-v2, etc.)
    - text2vec: Text2Vec embeddings (text2vec-base-en, etc.)
    - openclip: OpenClip embeddings (openclip-large-v2, etc.)
    - instructor: Instructor embeddings (hkunlp/instructor-large, etc.)
    - onnx: ONNX embeddings (onnx-large-v2, etc.)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the embedding service.

        Args:
            provider: The embedding provider to use
            model: The model name
            api_key: API key (if not provided, will look for environment variables)
            **kwargs: Additional configuration options
        """
        self.config = EmbeddingConfig(
            provider=provider,
            model=model,
            api_key=api_key or self._get_default_api_key(provider),
            **kwargs,
        )

        self._embedding_function = None
        self._initialize_embedding_function()

    @staticmethod
    def _get_default_api_key(provider: str) -> str | None:
        """Get default API key from environment variables."""
        env_key_map = {
            "azure": "AZURE_OPENAI_API_KEY",
            "amazon-bedrock": "AWS_ACCESS_KEY_ID",  # or AWS_PROFILE
            "cohere": "COHERE_API_KEY",
            "google-generativeai": "GOOGLE_API_KEY",
            "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
            "huggingface": "HUGGINGFACE_API_KEY",
            "jina": "JINA_API_KEY",
            "ollama": None,  # Ollama typically runs locally without API key
            "openai": "OPENAI_API_KEY",
            "roboflow": "ROBOFLOW_API_KEY",
            "voyageai": "VOYAGE_API_KEY",
            "watsonx": "WATSONX_API_KEY",
        }

        env_key = env_key_map.get(provider)
        if env_key:
            return os.getenv(env_key)
        return None

    def _initialize_embedding_function(self):
        """Initialize the embedding function using CrewAI's factory."""
        try:
            from crewai.rag.embeddings.factory import build_embedder

            # Build the configuration for CrewAI's factory
            config = self._build_provider_config()

            # Create the embedding function
            self._embedding_function = build_embedder(config)

            logger.info(
                f"Initialized {self.config.provider} embedding service with model "
                f"{self.config.model}"
            )

        except ImportError as e:
            raise ImportError(
                f"CrewAI embedding providers not available. "
                f"Make sure crewai is installed: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize embedding function: {e}")
            raise RuntimeError(
                f"Failed to initialize {self.config.provider} embedding service: {e}"
            ) from e

    def _build_provider_config(self) -> dict[str, Any]:
        """Build configuration dictionary for CrewAI's embedding factory."""
        base_config = {"provider": self.config.provider, "config": {}}

        # Provider-specific configuration mapping
        if self.config.provider == "openai":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "azure":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "voyageai":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model": self.config.model,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                **self.config.extra_config,
            }
        elif self.config.provider == "cohere":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider in ["google-generativeai", "google-vertex"]:
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "amazon-bedrock":
            base_config["config"] = {
                "aws_access_key_id": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "huggingface":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "jina":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "ollama":
            base_config["config"] = {
                "model": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "sentence-transformer":
            base_config["config"] = {
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "instructor":
            base_config["config"] = {
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "onnx":
            base_config["config"] = {
                **self.config.extra_config,
            }
        elif self.config.provider == "roboflow":
            base_config["config"] = {
                "api_key": self.config.api_key,
                **self.config.extra_config,
            }
        elif self.config.provider == "openclip":
            base_config["config"] = {
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "text2vec":
            base_config["config"] = {
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "watsonx":
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model_name": self.config.model,
                **self.config.extra_config,
            }
        elif self.config.provider == "custom":
            # Custom provider requires embedding_callable in extra_config
            base_config["config"] = {
                **self.config.extra_config,
            }
        else:
            # Generic configuration for any unlisted providers
            base_config["config"] = {
                "api_key": self.config.api_key,
                "model": self.config.model,
                **self.config.extra_config,
            }

        return base_config

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []

        try:
            # Use ChromaDB's embedding function interface
            embeddings = self._embedding_function([text])  # type: ignore
            return embeddings[0] if embeddings else []

        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            logger.warning("No valid texts provided for batch embedding")
            return []

        try:
            # Process in batches to avoid API limits
            all_embeddings = []

            for i in range(0, len(valid_texts), self.config.batch_size):
                batch = valid_texts[i : i + self.config.batch_size]
                batch_embeddings = self._embedding_function(batch)  # type: ignore
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise RuntimeError(f"Failed to generate batch embeddings: {e}") from e

    def get_embedding_dimension(self) -> int | None:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension or None if unknown
        """
        # Try to get dimension by generating a test embedding
        try:
            test_embedding = self.embed_text("test")
            return len(test_embedding) if test_embedding else None
        except Exception:
            logger.warning("Could not determine embedding dimension")
            return None

    def validate_connection(self) -> bool:
        """
        Validate that the embedding service is working correctly.

        Returns:
            True if the service is working, False otherwise
        """
        try:
            test_embedding = self.embed_text("test connection")
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False

    def get_service_info(self) -> dict[str, Any]:
        """
        Get information about the current embedding service.

        Returns:
            Dictionary with service information
        """
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "embedding_dimension": self.get_embedding_dimension(),
            "batch_size": self.config.batch_size,
            "is_connected": self.validate_connection(),
        }

    @classmethod
    def list_supported_providers(cls) -> list[str]:
        """
        List all supported embedding providers.

        Returns:
            List of supported provider names
        """
        return [
            "azure",
            "amazon-bedrock",
            "cohere",
            "custom",
            "google-generativeai",
            "google-vertex",
            "huggingface",
            "instructor",
            "jina",
            "ollama",
            "onnx",
            "openai",
            "openclip",
            "roboflow",
            "sentence-transformer",
            "text2vec",
            "voyageai",
            "watsonx",
        ]

    @classmethod
    def create_openai_service(
        cls,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create an OpenAI embedding service."""
        return cls(provider="openai", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_voyage_service(
        cls, model: str = "voyage-2", api_key: str | None = None, **kwargs: Any
    ) -> EmbeddingService:
        """Create a Voyage AI embedding service."""
        return cls(provider="voyageai", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_cohere_service(
        cls,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a Cohere embedding service."""
        return cls(provider="cohere", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_gemini_service(
        cls,
        model: str = "models/embedding-001",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a Google Gemini embedding service."""
        return cls(
            provider="google-generativeai", model=model, api_key=api_key, **kwargs
        )

    @classmethod
    def create_azure_service(
        cls,
        model: str = "text-embedding-ada-002",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create an Azure OpenAI embedding service."""
        return cls(provider="azure", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_bedrock_service(
        cls,
        model: str = "amazon.titan-embed-text-v1",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create an Amazon Bedrock embedding service."""
        return cls(provider="amazon-bedrock", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_huggingface_service(
        cls,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a Hugging Face embedding service."""
        return cls(provider="huggingface", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_sentence_transformer_service(
        cls,
        model: str = "all-MiniLM-L6-v2",
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a Sentence Transformers embedding service (local)."""
        return cls(provider="sentence-transformer", model=model, **kwargs)

    @classmethod
    def create_ollama_service(
        cls,
        model: str = "nomic-embed-text",
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create an Ollama embedding service (local)."""
        return cls(provider="ollama", model=model, **kwargs)

    @classmethod
    def create_jina_service(
        cls,
        model: str = "jina-embeddings-v2-base-en",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a Jina AI embedding service."""
        return cls(provider="jina", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_instructor_service(
        cls,
        model: str = "hkunlp/instructor-large",
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create an Instructor embedding service."""
        return cls(provider="instructor", model=model, **kwargs)

    @classmethod
    def create_watsonx_service(
        cls,
        model: str = "ibm/slate-125m-english-rtrvr",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a Watson X embedding service."""
        return cls(provider="watsonx", model=model, api_key=api_key, **kwargs)

    @classmethod
    def create_custom_service(
        cls,
        embedding_callable: Any,
        **kwargs: Any,
    ) -> EmbeddingService:
        """Create a custom embedding service with your own embedding function."""
        return cls(
            provider="custom",
            model="custom",
            extra_config={"embedding_callable": embedding_callable},
            **kwargs,
        )

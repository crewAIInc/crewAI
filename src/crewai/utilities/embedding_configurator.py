import os
from typing import Any, Dict, Optional, cast

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import validate_embedding_function


class EmbeddingConfigurator:
    def __init__(self):
        self.embedding_functions = {
            "openai": self._configure_openai,
            "azure": self._configure_azure,
            "ollama": self._configure_ollama,
            "vertexai": self._configure_vertexai,
            "google": self._configure_google,
            "cohere": self._configure_cohere,
            "voyageai": self._configure_voyageai,
            "bedrock": self._configure_bedrock,
            "huggingface": self._configure_huggingface,
            "watson": self._configure_watson,
            "custom": self._configure_custom,
        }

    def configure_embedder(
        self,
        embedder_config: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingFunction:
        """Configure and return an embedding function based on the provided config.
        
        Args:
            embedder_config: Optional configuration dictionary containing:
                - provider: Name of the embedding provider or EmbeddingFunction instance
                - config: Provider-specific configuration dictionary with options like:
                    - api_key: API key for the provider
                    - model: Model name to use for embeddings
                    - url: API endpoint URL (for some providers)
                    - session: Session object (for some providers)
        
        Returns:
            EmbeddingFunction: Configured embedding function for the specified provider
            
        Raises:
            ValueError: If custom embedding function is invalid
            Exception: If provider is not supported or configuration is invalid
            
        Examples:
            >>> config = {
            ...     "provider": "openai",
            ...     "config": {
            ...         "api_key": "your-api-key",
            ...         "model": "text-embedding-3-small"
            ...     }
            ... }
            >>> embedder = EmbeddingConfigurator().configure_embedder(config)
        """
        if embedder_config is None:
            return self._create_default_embedding_function()

        provider = embedder_config.get("provider")
        config = embedder_config.get("config", {})
        model_name = config.get("model") if provider != "custom" else None

        if provider not in self.embedding_functions:
            raise Exception(
                f"Unsupported embedding provider: {provider}, supported providers: {list(self.embedding_functions.keys())}"
            )

        embedding_function = self.embedding_functions[provider]
        return (
            embedding_function(config)
            if provider == "custom"
            else embedding_function(config, model_name)
        )

    @staticmethod
    def _create_default_embedding_function():
        """Create a default embedding function based on environment variables.
        
        Environment Variables:
            CREWAI_EMBEDDING_PROVIDER: The embedding provider to use (default: "openai")
            CREWAI_EMBEDDING_MODEL: The model to use for embeddings
            OPENAI_API_KEY: API key for OpenAI (required if using OpenAI provider)
        
        Returns:
            EmbeddingFunction: Configured embedding function
        """
        provider = os.getenv("CREWAI_EMBEDDING_PROVIDER", "openai")
        config = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("CREWAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
        return EmbeddingConfigurator().configure_embedder(
            {"provider": provider, "config": config}
        )

    @staticmethod
    def _configure_openai(config, model_name):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            api_base=config.get("api_base", None),
            api_type=config.get("api_type", None),
            api_version=config.get("api_version", None),
            default_headers=config.get("default_headers", None),
            dimensions=config.get("dimensions", None),
            deployment_id=config.get("deployment_id", None),
            organization_id=config.get("organization_id", None),
        )

    @staticmethod
    def _configure_azure(config, model_name):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=config.get("api_key"),
            api_base=config.get("api_base"),
            api_type=config.get("api_type", "azure"),
            api_version=config.get("api_version"),
            model_name=model_name,
            default_headers=config.get("default_headers"),
            dimensions=config.get("dimensions"),
            deployment_id=config.get("deployment_id"),
            organization_id=config.get("organization_id"),
        )

    @staticmethod
    def _configure_ollama(config, model_name):
        from chromadb.utils.embedding_functions.ollama_embedding_function import (
            OllamaEmbeddingFunction,
        )

        return OllamaEmbeddingFunction(
            url=config.get("url", "http://localhost:11434/api/embeddings"),
            model_name=model_name,
        )

    @staticmethod
    def _configure_vertexai(config, model_name):
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleVertexEmbeddingFunction,
        )

        return GoogleVertexEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
            project_id=config.get("project_id"),
            region=config.get("region"),
        )

    @staticmethod
    def _configure_google(config, model_name):
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleGenerativeAiEmbeddingFunction,
        )

        return GoogleGenerativeAiEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
            task_type=config.get("task_type"),
        )

    @staticmethod
    def _configure_cohere(config, model_name):
        from chromadb.utils.embedding_functions.cohere_embedding_function import (
            CohereEmbeddingFunction,
        )

        return CohereEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_voyageai(config, model_name):
        from chromadb.utils.embedding_functions.voyageai_embedding_function import (
            VoyageAIEmbeddingFunction,
        )

        return VoyageAIEmbeddingFunction(
            model_name=model_name,
            api_key=config.get("api_key"),
        )

    @staticmethod
    def _configure_bedrock(config, model_name):
        from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
            AmazonBedrockEmbeddingFunction,
        )

        # Allow custom model_name override with backwards compatibility
        kwargs = {"session": config.get("session")}
        if model_name is not None:
            kwargs["model_name"] = model_name
        return AmazonBedrockEmbeddingFunction(**kwargs)

    @staticmethod
    def _configure_huggingface(config, model_name):
        from chromadb.utils.embedding_functions.huggingface_embedding_function import (
            HuggingFaceEmbeddingServer,
        )

        return HuggingFaceEmbeddingServer(
            url=config.get("api_url"),
        )

    @staticmethod
    def _configure_custom(config, model_name):
        """Configure a custom embedding function.
        
        Args:
            config: Configuration dictionary containing:
                - embedder: Custom EmbeddingFunction instance
            model_name: Not used for custom embedders
            
        Returns:
            EmbeddingFunction: The validated custom embedding function
            
        Raises:
            ValueError: If embedder is missing or invalid
        """
        embedder = config.get("embedder")
        if not embedder or not isinstance(embedder, EmbeddingFunction):
            raise ValueError("Custom provider requires a valid EmbeddingFunction instance")
        
        try:
            validate_embedding_function(embedder)
            return embedder
        except Exception as e:
            raise ValueError(f"Invalid custom embedding function: {str(e)}")

    @staticmethod
    def _configure_watson(config, model_name):
        try:
            import ibm_watsonx_ai.foundation_models as watson_models
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
        except ImportError as e:
            raise ImportError(
                "IBM Watson dependencies are not installed. Please install them to use Watson embedding."
            ) from e

        class WatsonEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                if isinstance(input, str):
                    input = [input]

                embed_params = {
                    EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
                    EmbedParams.RETURN_OPTIONS: {"input_text": True},
                }

                embedding = watson_models.Embeddings(
                    model_id=config.get("model"),
                    params=embed_params,
                    credentials=Credentials(
                        api_key=config.get("api_key"), url=config.get("api_url")
                    ),
                    project_id=config.get("project_id"),
                )

                try:
                    embeddings = embedding.embed_documents(input)
                    return cast(Embeddings, embeddings)
                except Exception as e:
                    print("Error during Watson embedding:", e)
                    raise e

        return WatsonEmbeddingFunction()

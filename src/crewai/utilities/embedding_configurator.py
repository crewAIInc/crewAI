import os
from typing import Any, Callable, Literal, cast

from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.types import validate_embedding_function
from pydantic import BaseModel


class EmbeddingProviderConfig(BaseModel):
    """Configuration model for embedding providers.

    Attributes:
        # Core Model Configuration
        model (str | None): The model identifier for embeddings, used across multiple providers
            like OpenAI, Azure, Watson, etc.
        embedder (str | Callable | None): Custom embedding function or callable for custom
            embedding implementations.

        # API Authentication & Configuration
        api_key (str | None): Authentication key for various providers (OpenAI, VertexAI,
            Google, Cohere, VoyageAI, Watson).
        api_base (str | None): Base API URL override for OpenAI and Azure services.
        api_type (str | None): API type specification, particularly used for Azure configuration.
        api_version (str | None): API version for OpenAI and Azure services.
        api_url (str | None): API endpoint URL, used by HuggingFace and Watson services.
        url (str | None): Base URL for the embedding service, primarily used for Ollama and
            HuggingFace endpoints.

        # Service-Specific Configuration
        project_id (str | None): Project identifier used by VertexAI and Watson services.
        organization_id (str | None): Organization identifier for OpenAI and Azure services.
        deployment_id (str | None): Deployment identifier for OpenAI and Azure services.
        region (str | None): Geographic region for VertexAI services.
        session (str | None): Session configuration for Amazon Bedrock embeddings.

        # Request Configuration
        task_type (str | None): Specifies the task type for Google Generative AI embeddings.
        default_headers (str | None): Custom headers for OpenAI and Azure API requests.
        dimensions (str | None): Output dimensions specification for OpenAI and Azure embeddings.
    """

    # Core Model Configuration
    model: str | None = None
    embedder: str | Callable | None = None

    # API Authentication & Configuration
    api_key: str | None = None
    api_base: str | None = None
    api_type: str | None = None
    api_version: str | None = None
    api_url: str | None = None
    url: str | None = None

    # Service-Specific Configuration
    project_id: str | None = None
    organization_id: str | None = None
    deployment_id: str | None = None
    region: str | None = None
    session: str | None = None

    # Request Configuration
    task_type: str | None = None
    default_headers: str | None = None
    dimensions: str | None = None


class EmbeddingConfig(BaseModel):
    provider: Literal[
        "openai",
        "azure",
        "ollama",
        "vertexai",
        "google",
        "cohere",
        "voyageai",
        "bedrock",
        "huggingface",
        "watson",
        "custom",
    ]
    config: EmbeddingProviderConfig | None = None


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
        embedder_config: EmbeddingConfig | None = None,
    ) -> EmbeddingFunction:
        """Configures and returns an embedding function based on the provided config."""
        if embedder_config is None:
            return self._create_default_embedding_function()

        provider = embedder_config.provider
        config = (
            embedder_config.config
            if embedder_config.config
            else EmbeddingProviderConfig()
        )
        model_name = config.model if provider != "custom" else None

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
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    @staticmethod
    def _configure_openai(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            model_name=model_name,
            api_base=config.api_base,
            api_type=config.api_type,
            api_version=config.api_version,
            default_headers=config.default_headers,
            dimensions=config.dimensions,
            deployment_id=config.deployment_id,
            organization_id=config.organization_id,
        )

    @staticmethod
    def _configure_azure(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=config.api_key,
            api_base=config.api_base,
            api_type=config.api_type if config.api_type else "azure",
            api_version=config.api_version,
            model_name=model_name,
            default_headers=config.default_headers,
            dimensions=config.dimensions,
            deployment_id=config.deployment_id,
            organization_id=config.organization_id,
        )

    @staticmethod
    def _configure_ollama(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.ollama_embedding_function import (
            OllamaEmbeddingFunction,
        )

        return OllamaEmbeddingFunction(
            url=config.url if config.url else "http://localhost:11434/api/embeddings",
            model_name=model_name,
        )

    @staticmethod
    def _configure_vertexai(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleVertexEmbeddingFunction,
        )

        return GoogleVertexEmbeddingFunction(
            model_name=model_name,
            api_key=config.api_key,
            project_id=config.project_id,
            region=config.region,
        )

    @staticmethod
    def _configure_google(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.google_embedding_function import (
            GoogleGenerativeAiEmbeddingFunction,
        )

        return GoogleGenerativeAiEmbeddingFunction(
            model_name=model_name,
            api_key=config.api_key,
            task_type=config.task_type,
        )

    @staticmethod
    def _configure_cohere(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.cohere_embedding_function import (
            CohereEmbeddingFunction,
        )

        return CohereEmbeddingFunction(
            model_name=model_name,
            api_key=config.api_key,
        )

    @staticmethod
    def _configure_voyageai(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.voyageai_embedding_function import (
            VoyageAIEmbeddingFunction,
        )

        return VoyageAIEmbeddingFunction(
            model_name=model_name,
            api_key=config.api_key,
        )

    @staticmethod
    def _configure_bedrock(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
            AmazonBedrockEmbeddingFunction,
        )

        # Allow custom model_name override with backwards compatibility
        kwargs = {"session": config.session}
        if model_name is not None:
            kwargs["model_name"] = model_name
        return AmazonBedrockEmbeddingFunction(**kwargs)

    @staticmethod
    def _configure_huggingface(config: EmbeddingProviderConfig, model_name: str):
        from chromadb.utils.embedding_functions.huggingface_embedding_function import (
            HuggingFaceEmbeddingServer,
        )

        return HuggingFaceEmbeddingServer(
            url=config.api_url,
        )

    @staticmethod
    def _configure_watson(config: EmbeddingProviderConfig, model_name: str):
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
                    model_id=config.model,
                    params=embed_params,
                    credentials=Credentials(api_key=config.api_key, url=config.api_url),
                    project_id=config.project_id,
                )

                try:
                    embeddings = embedding.embed_documents(input)
                    return cast(Embeddings, embeddings)
                except Exception as e:
                    print("Error during Watson embedding:", e)
                    raise e

        return WatsonEmbeddingFunction()

    @staticmethod
    def _configure_custom(config: EmbeddingProviderConfig):
        custom_embedder = config.embedder
        if isinstance(custom_embedder, EmbeddingFunction):
            try:
                validate_embedding_function(custom_embedder)
                return custom_embedder
            except Exception as e:
                raise ValueError(f"Invalid custom embedding function: {str(e)}")
        elif callable(custom_embedder):
            try:
                instance = custom_embedder()
                if isinstance(instance, EmbeddingFunction):
                    validate_embedding_function(instance)
                    return instance
                raise ValueError(
                    "Custom embedder does not create an EmbeddingFunction instance"
                )
            except Exception as e:
                raise ValueError(f"Error instantiating custom embedder: {str(e)}")
        else:
            raise ValueError(
                "Custom embedder must be an instance of `EmbeddingFunction` or a callable that creates one"
            )

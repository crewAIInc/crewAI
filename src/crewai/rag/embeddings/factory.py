"""Minimal embedding function factory for CrewAI."""

import os
from collections.abc import Callable, MutableMapping
from typing import Any, Final, Literal, TypedDict

from chromadb import EmbeddingFunction
from chromadb.utils.embedding_functions.amazon_bedrock_embedding_function import (
    AmazonBedrockEmbeddingFunction,
)
from chromadb.utils.embedding_functions.cohere_embedding_function import (
    CohereEmbeddingFunction,
)
from chromadb.utils.embedding_functions.google_embedding_function import (
    GoogleGenerativeAiEmbeddingFunction,
    GooglePalmEmbeddingFunction,
    GoogleVertexEmbeddingFunction,
)
from chromadb.utils.embedding_functions.huggingface_embedding_function import (
    HuggingFaceEmbeddingFunction,
)
from chromadb.utils.embedding_functions.instructor_embedding_function import (
    InstructorEmbeddingFunction,
)
from chromadb.utils.embedding_functions.jina_embedding_function import (
    JinaEmbeddingFunction,
)
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from chromadb.utils.embedding_functions.open_clip_embedding_function import (
    OpenCLIPEmbeddingFunction,
)
from chromadb.utils.embedding_functions.openai_embedding_function import (
    OpenAIEmbeddingFunction,
)
from chromadb.utils.embedding_functions.roboflow_embedding_function import (
    RoboflowEmbeddingFunction,
)
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from chromadb.utils.embedding_functions.text2vec_embedding_function import (
    Text2VecEmbeddingFunction,
)
from typing_extensions import NotRequired

from crewai.rag.embeddings.types import EmbeddingOptions

AllowedEmbeddingProviders = Literal[
    "openai",
    "cohere",
    "ollama",
    "huggingface",
    "sentence-transformer",
    "instructor",
    "google-palm",
    "google-generativeai",
    "google-vertex",
    "amazon-bedrock",
    "jina",
    "roboflow",
    "openclip",
    "text2vec",
    "onnx",
]


class EmbedderConfig(TypedDict):
    """Configuration for embedding functions with nested format."""

    provider: AllowedEmbeddingProviders
    config: NotRequired[dict[str, Any]]


EMBEDDING_PROVIDERS: Final[
    dict[AllowedEmbeddingProviders, Callable[..., EmbeddingFunction]]
] = {
    "openai": OpenAIEmbeddingFunction,
    "cohere": CohereEmbeddingFunction,
    "ollama": OllamaEmbeddingFunction,
    "huggingface": HuggingFaceEmbeddingFunction,
    "sentence-transformer": SentenceTransformerEmbeddingFunction,
    "instructor": InstructorEmbeddingFunction,
    "google-palm": GooglePalmEmbeddingFunction,
    "google-generativeai": GoogleGenerativeAiEmbeddingFunction,
    "google-vertex": GoogleVertexEmbeddingFunction,
    "amazon-bedrock": AmazonBedrockEmbeddingFunction,
    "jina": JinaEmbeddingFunction,
    "roboflow": RoboflowEmbeddingFunction,
    "openclip": OpenCLIPEmbeddingFunction,
    "text2vec": Text2VecEmbeddingFunction,
    "onnx": ONNXMiniLM_L6_V2,
}

PROVIDER_ENV_MAPPING: Final[dict[AllowedEmbeddingProviders, tuple[str, str]]] = {
    "openai": ("OPENAI_API_KEY", "api_key"),
    "cohere": ("COHERE_API_KEY", "api_key"),
    "huggingface": ("HUGGINGFACE_API_KEY", "api_key"),
    "google-palm": ("GOOGLE_API_KEY", "api_key"),
    "google-generativeai": ("GOOGLE_API_KEY", "api_key"),
    "google-vertex": ("GOOGLE_API_KEY", "api_key"),
    "jina": ("JINA_API_KEY", "api_key"),
    "roboflow": ("ROBOFLOW_API_KEY", "api_key"),
}


def _inject_api_key_from_env(
    provider: AllowedEmbeddingProviders, config_dict: MutableMapping[str, Any]
) -> None:
    """Inject API key or other required configuration from environment if not explicitly provided.

    Args:
        provider: The embedding provider name
        config_dict: The configuration dictionary to modify in-place

    Raises:
        ImportError: If required libraries for certain providers are not installed
        ValueError: If AWS session creation fails for amazon-bedrock
    """
    if provider in PROVIDER_ENV_MAPPING:
        env_var_name, config_key = PROVIDER_ENV_MAPPING[provider]
        if config_key not in config_dict:
            env_value = os.getenv(env_var_name)
            if env_value:
                config_dict[config_key] = env_value

    if provider == "amazon-bedrock":
        if "session" not in config_dict:
            try:
                import boto3  # type: ignore[import]

                config_dict["session"] = boto3.Session()
            except ImportError as e:
                raise ImportError(
                    "boto3 is required for amazon-bedrock embeddings. "
                    "Install it with: uv add boto3"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Failed to create AWS session for amazon-bedrock. "
                    f"Ensure AWS credentials are configured. Error: {e}"
                ) from e


def get_embedding_function(
    config: EmbeddingOptions | EmbedderConfig | None = None,
) -> EmbeddingFunction:
    """Get embedding function - delegates to ChromaDB.

    Args:
        config: Optional configuration - either:
            - EmbeddingOptions: Pydantic model with flat configuration
            - EmbedderConfig: TypedDict with nested format {"provider": str, "config": dict}
            - None: Uses default OpenAI configuration

    Returns:
        EmbeddingFunction instance ready for use with ChromaDB

    Supported providers:
        - openai: OpenAI embeddings
        - cohere: Cohere embeddings
        - ollama: Ollama local embeddings
        - huggingface: HuggingFace embeddings
        - sentence-transformer: Local sentence transformers
        - instructor: Instructor embeddings for specialized tasks
        - google-palm: Google PaLM embeddings
        - google-generativeai: Google Generative AI embeddings
        - google-vertex: Google Vertex AI embeddings
        - amazon-bedrock: AWS Bedrock embeddings
        - jina: Jina AI embeddings
        - roboflow: Roboflow embeddings for vision tasks
        - openclip: OpenCLIP embeddings for multimodal tasks
        - text2vec: Text2Vec embeddings
        - onnx: ONNX MiniLM-L6-v2 (no API key needed, included with ChromaDB)

    Examples:
        # Use default OpenAI embedding
        >>> embedder = get_embedding_function()

        # Use Cohere with dict
        >>> embedder = get_embedding_function(EmbedderConfig(**{
        ...     "provider": "cohere",
        ...     "config": {
        ...         "api_key": "your-key",
        ...         "model_name": "embed-english-v3.0"
        ...     }
        ... }))

        # Use with EmbeddingOptions
        >>> embedder = get_embedding_function(
        ...     EmbeddingOptions(provider="sentence-transformer", model_name="all-MiniLM-L6-v2")
        ... )

        # Use Azure OpenAI
        >>> embedder = get_embedding_function(EmbedderConfig(**{
        ...     "provider": "openai",
        ...     "config": {
        ...         "api_key": "your-azure-key",
        ...         "api_base": "https://your-resource.openai.azure.com/",
        ...         "api_type": "azure",
        ...         "api_version": "2023-05-15",
        ...         "model": "text-embedding-3-small",
        ...         "deployment_id": "your-deployment-name"
        ...     }
        ... })

        >>> embedder = get_embedding_function(EmbedderConfig(**{
        ...     "provider": "onnx"
        ... })
    """
    if config is None:
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    provider: AllowedEmbeddingProviders
    config_dict: dict[str, Any]

    if isinstance(config, EmbeddingOptions):
        config_dict = config.model_dump(exclude_none=True)
        provider = config_dict["provider"]
    else:
        provider = config["provider"]
        nested: dict[str, Any] = config.get("config", {})

        if not nested and len(config) > 1:
            raise ValueError(
                "Invalid embedder configuration format. "
                "Configuration must be nested under a 'config' key. "
                "Example: {'provider': 'openai', 'config': {'api_key': '...', 'model': '...'}}"
            )

        config_dict = dict(nested)
        if "model" in config_dict and "model_name" not in config_dict:
            config_dict["model_name"] = config_dict.pop("model")

    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Available providers: {list(EMBEDDING_PROVIDERS.keys())}"
        )

    _inject_api_key_from_env(provider, config_dict)

    return EMBEDDING_PROVIDERS[provider](**config_dict)

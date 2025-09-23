"""Minimal embedding function factory for CrewAI."""

import os

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

from crewai.rag.embeddings.types import EmbeddingOptions


def _create_watson_embedding_function(**config_dict) -> EmbeddingFunction:
    """Create Watson embedding function with proper error handling."""
    try:
        import ibm_watsonx_ai.foundation_models as watson_models  # type: ignore[import-not-found]
        from ibm_watsonx_ai import Credentials  # type: ignore[import-not-found]
        from ibm_watsonx_ai.metanames import (  # type: ignore[import-not-found]
            EmbedTextParamsMetaNames as EmbedParams,
        )
    except ImportError as e:
        raise ImportError(
            "IBM Watson dependencies are not installed. Please install them to use Watson embedding."
        ) from e

    class WatsonEmbeddingFunction(EmbeddingFunction):
        def __init__(self, **kwargs):
            self.config = kwargs

        def __call__(self, input):
            if isinstance(input, str):
                input = [input]

            embed_params = {
                EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
                EmbedParams.RETURN_OPTIONS: {"input_text": True},
            }

            embedding = watson_models.Embeddings(
                model_id=self.config.get("model_name") or self.config.get("model"),
                params=embed_params,
                credentials=Credentials(
                    api_key=self.config.get("api_key"), 
                    url=self.config.get("api_url") or self.config.get("url")
                ),
                project_id=self.config.get("project_id"),
            )

            try:
                embeddings = embedding.embed_documents(input)
                return embeddings
            except Exception as e:
                raise RuntimeError(f"Error during Watson embedding: {e}") from e

    return WatsonEmbeddingFunction(**config_dict)


def get_embedding_function(
    config: EmbeddingOptions | dict | None = None,
) -> EmbeddingFunction:
    """Get embedding function - delegates to ChromaDB.

    Args:
        config: Optional configuration - either an EmbeddingOptions object or a dict with:
            - provider: The embedding provider to use (default: "openai")
            - Any other provider-specific parameters

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
        - watson: IBM Watson embeddings

    Examples:
        # Use default OpenAI embedding
        >>> embedder = get_embedding_function()

        # Use Cohere with dict
        >>> embedder = get_embedding_function({
        ...     "provider": "cohere",
        ...     "api_key": "your-key",
        ...     "model_name": "embed-english-v3.0"
        ... })

        # Use with EmbeddingOptions
        >>> embedder = get_embedding_function(
        ...     EmbeddingOptions(provider="sentence-transformer", model_name="all-MiniLM-L6-v2")
        ... )

        # Use local sentence transformers (no API key needed)
        >>> embedder = get_embedding_function({
        ...     "provider": "sentence-transformer",
        ...     "model_name": "all-MiniLM-L6-v2"
        ... })

        # Use Ollama for local embeddings
        >>> embedder = get_embedding_function({
        ...     "provider": "ollama",
        ...     "model_name": "nomic-embed-text"
        ... })

        # Use ONNX (no API key needed)
        >>> embedder = get_embedding_function({
        ...     "provider": "onnx"
        ... })

        # Use Watson embeddings
        >>> embedder = get_embedding_function({
        ...     "provider": "watson",
        ...     "api_key": "your-watson-api-key",
        ...     "api_url": "your-watson-url",
        ...     "project_id": "your-project-id",
        ...     "model_name": "ibm/slate-125m-english-rtrvr"
        ... })
    """
    if config is None:
        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    # Handle EmbeddingOptions object
    if isinstance(config, EmbeddingOptions):
        config_dict = config.model_dump(exclude_none=True)
    else:
        config_dict = config.copy()

    provider = config_dict.pop("provider", "openai")

    embedding_functions = {
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
        "watson": _create_watson_embedding_function,
    }

    if provider not in embedding_functions:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Available providers: {list(embedding_functions.keys())}"
        )
    return embedding_functions[provider](**config_dict)

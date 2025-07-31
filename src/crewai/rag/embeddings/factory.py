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
    GooglePalmEmbeddingFunction,
    GoogleGenerativeAiEmbeddingFunction,
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
        - openai: OpenAI embeddings (default)
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
        # Use default OpenAI with retry logic
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
    }

    if provider not in embedding_functions:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Available providers: {list(embedding_functions.keys())}"
        )
    return embedding_functions[provider](**config_dict)

"""Google Generative AI embeddings provider."""

from chromadb.utils.embedding_functions.google_embedding_function import (
    GoogleGenerativeAiEmbeddingFunction,
)
from pydantic import Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class GenerativeAiProvider(BaseEmbeddingsProvider[GoogleGenerativeAiEmbeddingFunction]):
    """Google Generative AI embeddings provider."""

    embedding_callable: type[GoogleGenerativeAiEmbeddingFunction] = Field(
        default=GoogleGenerativeAiEmbeddingFunction,
        description="Google Generative AI embedding function class",
    )
    model_name: str = Field(
        default="models/embedding-001",
        description="Model name to use for embeddings",
        validation_alias="EMBEDDINGS_GOOGLE_GENERATIVE_AI_MODEL_NAME",
    )
    api_key: str = Field(
        description="Google API key", validation_alias="EMBEDDINGS_GOOGLE_API_KEY"
    )
    task_type: str = Field(
        default="RETRIEVAL_DOCUMENT",
        description="Task type for embeddings",
        validation_alias="EMBEDDINGS_GOOGLE_GENERATIVE_AI_TASK_TYPE",
    )

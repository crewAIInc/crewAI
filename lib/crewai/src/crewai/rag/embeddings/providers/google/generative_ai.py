"""Google Generative AI embeddings provider."""

from typing import Literal

from chromadb.utils.embedding_functions.google_embedding_function import (
    GoogleGenerativeAiEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class GenerativeAiProvider(BaseEmbeddingsProvider[GoogleGenerativeAiEmbeddingFunction]):
    """Google Generative AI embeddings provider."""

    embedding_callable: type[GoogleGenerativeAiEmbeddingFunction] = Field(
        default=GoogleGenerativeAiEmbeddingFunction,
        description="Google Generative AI embedding function class",
    )
    model_name: Literal[
        "gemini-embedding-001", "text-embedding-005", "text-multilingual-embedding-002"
    ] = Field(
        default="gemini-embedding-001",
        description="Model name to use for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_GENERATIVE_AI_MODEL_NAME", "model"
        ),
    )
    api_key: str = Field(
        description="Google API key",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"
        ),
    )
    task_type: str = Field(
        default="RETRIEVAL_DOCUMENT",
        description="Task type for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_GOOGLE_GENERATIVE_AI_TASK_TYPE",
            "GOOGLE_GENERATIVE_AI_TASK_TYPE",
            "GEMINI_TASK_TYPE",
        ),
    )

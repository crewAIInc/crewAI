"""Google Generative AI embeddings provider."""

from typing import Literal

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class GenerativeAiProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Google Generative AI embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "google_embedding_function", "GoogleGenerativeAiEmbeddingFunction"
        ),
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

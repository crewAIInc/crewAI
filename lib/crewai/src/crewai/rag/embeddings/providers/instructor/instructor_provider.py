"""Instructor embeddings provider."""

from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.chromadb_loader import load_chromadb_embedding_function


class InstructorProvider(BaseEmbeddingsProvider[EmbeddingFunction]):  # type: ignore[type-arg]
    """Instructor embeddings provider."""

    embedding_callable: type[EmbeddingFunction] = Field(  # type: ignore[type-arg]
        default_factory=lambda: load_chromadb_embedding_function(
            "instructor_embedding_function", "InstructorEmbeddingFunction"
        ),
        description="Instructor embedding function class",
    )
    model_name: str = Field(
        default="hkunlp/instructor-base",
        description="Model name to use",
        validation_alias=AliasChoices(
            "EMBEDDINGS_INSTRUCTOR_MODEL_NAME",
            "INSTRUCTOR_MODEL_NAME",
            "model",
        ),
    )
    device: str = Field(
        default="cpu",
        description="Device to run model on (e.g., cpu, cuda, mps, xpu)",
        validation_alias=AliasChoices(
            "EMBEDDINGS_INSTRUCTOR_DEVICE", "INSTRUCTOR_DEVICE"
        ),
    )
    instruction: str | None = Field(
        default=None,
        description="Instruction for embeddings",
        validation_alias=AliasChoices(
            "EMBEDDINGS_INSTRUCTOR_INSTRUCTION", "INSTRUCTOR_INSTRUCTION"
        ),
    )

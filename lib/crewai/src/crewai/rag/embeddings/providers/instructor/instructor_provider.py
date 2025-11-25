"""Instructor embeddings provider."""

from chromadb.utils.embedding_functions.instructor_embedding_function import (
    InstructorEmbeddingFunction,
)
from pydantic import AliasChoices, Field

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider


class InstructorProvider(BaseEmbeddingsProvider[InstructorEmbeddingFunction]):
    """Instructor embeddings provider."""

    embedding_callable: type[InstructorEmbeddingFunction] = Field(
        default=InstructorEmbeddingFunction,
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
        description="Device to run model on (cpu or cuda)",
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

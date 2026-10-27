"""Instructor embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.instructor.types import (
    InstructorProviderConfig,
    InstructorProviderSpec,
)


__all__ = [
    "InstructorProvider",
    "InstructorProviderConfig",
    "InstructorProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(
        name, {"InstructorProvider": f"{__name__}.instructor_provider"}
    )

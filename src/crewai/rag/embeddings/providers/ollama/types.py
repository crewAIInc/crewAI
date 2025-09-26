"""Type definitions for Ollama embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class OllamaProviderConfig(TypedDict, total=False):
    """Configuration for Ollama provider."""

    url: Annotated[str, "http://localhost:11434/api/embeddings"]
    model_name: str


class OllamaProviderSpec(TypedDict, total=False):
    """Ollama provider specification."""

    provider: Required[Literal["ollama"]]
    config: OllamaProviderConfig

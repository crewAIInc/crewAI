"""Base classes for missing provider configurations."""

from dataclasses import field
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pyd_dataclass


def missing_provider_error(provider: str) -> RuntimeError:
    """Create the error raised when an optional RAG provider is unavailable."""
    return RuntimeError(
        f"provider '{provider}' requested but not installed. "
        f"Install the extra: `uv add 'crewai[{provider}]'`."
    )


@pyd_dataclass(config=ConfigDict(extra="forbid"))
class _MissingProvider:
    """Base class for missing provider configurations.

    Raises RuntimeError when instantiated to indicate missing dependencies.
    """

    provider: Literal["chromadb", "qdrant", "__missing__"] = field(
        default="__missing__"
    )

    def __post_init__(self) -> None:
        """Raises error indicating the provider is not installed."""
        raise missing_provider_error(self.provider)

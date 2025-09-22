"""Base classes for missing provider configurations."""

from dataclasses import field
from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pyd_dataclass


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
        raise RuntimeError(
            f"provider '{self.provider}' requested but not installed. "
            f"Install the extra: `uv add crewai'[{self.provider}]'`."
        )

"""Roboflow embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.roboflow.types import (
    RoboflowProviderConfig,
    RoboflowProviderSpec,
)


__all__ = [
    "RoboflowProvider",
    "RoboflowProviderConfig",
    "RoboflowProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"RoboflowProvider": f"{__name__}.roboflow_provider"})

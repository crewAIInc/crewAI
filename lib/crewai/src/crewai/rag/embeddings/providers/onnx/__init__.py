"""ONNX embedding providers."""

from crewai.rag.embeddings.providers._lazy import load_provider
from crewai.rag.embeddings.providers.onnx.types import (
    ONNXProviderConfig,
    ONNXProviderSpec,
)


__all__ = [
    "ONNXProvider",
    "ONNXProviderConfig",
    "ONNXProviderSpec",
]


def __getattr__(name: str) -> object:
    return load_provider(name, {"ONNXProvider": f"{__name__}.onnx_provider"})

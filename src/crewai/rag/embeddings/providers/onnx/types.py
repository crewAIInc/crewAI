"""Type definitions for ONNX embedding providers."""

from typing import Literal, TypedDict


class ONNXProviderConfig(TypedDict, total=False):
    """Configuration for ONNX provider."""

    preferred_providers: list[str]


class ONNXProviderSpec(TypedDict):
    """ONNX provider specification."""

    provider: Literal["onnx"]
    config: ONNXProviderConfig

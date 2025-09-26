"""Type definitions for ONNX embedding providers."""

from typing import Literal, TypedDict

from typing_extensions import Required


class ONNXProviderConfig(TypedDict, total=False):
    """Configuration for ONNX provider."""

    preferred_providers: list[str]


class ONNXProviderSpec(TypedDict, total=False):
    """ONNX provider specification."""

    provider: Required[Literal["onnx"]]
    config: ONNXProviderConfig

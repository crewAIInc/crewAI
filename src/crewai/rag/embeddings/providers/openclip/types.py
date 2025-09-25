"""Type definitions for OpenCLIP embedding providers."""

from typing import Annotated, Literal, TypedDict


class OpenCLIPProviderConfig(TypedDict, total=False):
    """Configuration for OpenCLIP provider."""

    model_name: Annotated[str, "ViT-B-32"]
    checkpoint: Annotated[str, "laion2b_s34b_b79k"]
    device: Annotated[str, "cpu"]


class OpenCLIPProviderSpec(TypedDict):
    """OpenCLIP provider specification."""

    provider: Literal["openclip"]
    config: OpenCLIPProviderConfig

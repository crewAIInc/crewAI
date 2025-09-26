"""Type definitions for OpenCLIP embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class OpenCLIPProviderConfig(TypedDict, total=False):
    """Configuration for OpenCLIP provider."""

    model_name: Annotated[str, "ViT-B-32"]
    checkpoint: Annotated[str, "laion2b_s34b_b79k"]
    device: Annotated[str, "cpu"]


class OpenCLIPProviderSpec(TypedDict):
    """OpenCLIP provider specification."""

    provider: Required[Literal["openclip"]]
    config: OpenCLIPProviderConfig

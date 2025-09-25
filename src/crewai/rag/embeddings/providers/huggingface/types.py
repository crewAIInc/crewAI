"""Type definitions for HuggingFace embedding providers."""

from typing import Literal, TypedDict


class HuggingFaceProviderConfig(TypedDict, total=False):
    """Configuration for HuggingFace provider."""

    url: str


class HuggingFaceProviderSpec(TypedDict):
    """HuggingFace provider specification."""

    provider: Literal["huggingface"]
    config: HuggingFaceProviderConfig

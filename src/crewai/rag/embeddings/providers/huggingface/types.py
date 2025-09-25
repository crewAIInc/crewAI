"""Type definitions for HuggingFace embedding providers."""

from typing import Literal, Required, TypedDict


class HuggingFaceProviderConfig(TypedDict, total=False):
    """Configuration for HuggingFace provider."""

    url: Required[str]


class HuggingFaceProviderSpec(TypedDict):
    """HuggingFace provider specification."""

    provider: Literal["huggingface"]
    config: HuggingFaceProviderConfig

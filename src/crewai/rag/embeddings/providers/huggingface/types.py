"""Type definitions for HuggingFace embedding providers."""

from typing import Literal, TypedDict

from typing_extensions import Required


class HuggingFaceProviderConfig(TypedDict, total=False):
    """Configuration for HuggingFace provider."""

    url: str


class HuggingFaceProviderSpec(TypedDict, total=False):
    """HuggingFace provider specification."""

    provider: Required[Literal["huggingface"]]
    config: HuggingFaceProviderConfig

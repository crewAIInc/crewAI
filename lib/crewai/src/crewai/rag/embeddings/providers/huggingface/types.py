"""Type definitions for HuggingFace embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class HuggingFaceProviderConfig(TypedDict, total=False):
    """Configuration for HuggingFace provider."""

    api_key: str
    model: Annotated[
        str, "sentence-transformers/all-MiniLM-L6-v2"
    ]  # alias for model_name for backward compat
    model_name: Annotated[str, "sentence-transformers/all-MiniLM-L6-v2"]


class HuggingFaceProviderSpec(TypedDict, total=False):
    """HuggingFace provider specification."""

    provider: Required[Literal["huggingface"]]
    config: HuggingFaceProviderConfig

"""Type definitions for VoyageAI embedding providers."""

from typing import Literal

from typing_extensions import Required, TypedDict


class VoyageAIProviderConfig(TypedDict, total=False):
    """Configuration for VoyageAI provider."""

    api_key: Required[str]
    model: Required[str]
    input_type: str
    truncation: bool
    output_dtype: str
    output_dimension: int
    max_retries: int
    timeout: float


class VoyageAIProviderSpec(TypedDict):
    """VoyageAI provider specification."""

    provider: Required[Literal["voyageai"]]
    config: VoyageAIProviderConfig

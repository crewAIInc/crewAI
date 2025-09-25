"""Type definitions for VoyageAI embedding providers."""

from typing import Annotated, Literal, TypedDict


class VoyageAIProviderConfig(TypedDict, total=False):
    """Configuration for VoyageAI provider."""

    api_key: str
    model: Annotated[str, "voyage-2"]
    input_type: str
    truncation: Annotated[bool, True]
    output_dtype: str
    output_dimension: int
    max_retries: Annotated[int, 0]
    timeout: float


class VoyageAIProviderSpec(TypedDict):
    """VoyageAI provider specification."""

    provider: Literal["voyageai"]
    config: VoyageAIProviderConfig

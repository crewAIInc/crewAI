"""Type definitions for Cohere embedding providers."""

from typing import Annotated, Literal, TypedDict


class CohereProviderConfig(TypedDict, total=False):
    """Configuration for Cohere provider."""

    api_key: str
    model_name: Annotated[str, "large"]


class CohereProviderSpec(TypedDict):
    """Cohere provider specification."""

    provider: Literal["cohere"]
    config: CohereProviderConfig

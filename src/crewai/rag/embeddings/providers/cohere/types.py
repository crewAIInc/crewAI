"""Type definitions for Cohere embedding providers."""

from typing import Annotated, Literal, Required, TypedDict


class CohereProviderConfig(TypedDict, total=False):
    """Configuration for Cohere provider."""

    api_key: Required[str]
    model_name: Annotated[str, "large"]


class CohereProviderSpec(TypedDict):
    """Cohere provider specification."""

    provider: Literal["cohere"]
    config: CohereProviderConfig

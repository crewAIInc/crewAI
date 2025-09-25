"""Type definitions for Cohere embedding providers."""

from typing import Annotated, Literal, TypedDict

from typing_extensions import Required


class CohereProviderConfig(TypedDict, total=False):
    """Configuration for Cohere provider."""

    api_key: str
    model_name: Annotated[str, "large"]


class CohereProviderSpec(TypedDict, total=False):
    """Cohere provider specification."""

    provider: Required[Literal["cohere"]]
    config: CohereProviderConfig

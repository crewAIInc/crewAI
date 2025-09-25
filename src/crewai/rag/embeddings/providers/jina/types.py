"""Type definitions for Jina embedding providers."""

from typing import Annotated, Literal, Required, TypedDict


class JinaProviderConfig(TypedDict, total=False):
    """Configuration for Jina provider."""

    api_key: Required[str]
    model_name: Annotated[str, "jina-embeddings-v2-base-en"]


class JinaProviderSpec(TypedDict):
    """Jina provider specification."""

    provider: Literal["jina"]
    config: JinaProviderConfig

"""Type definitions for Jina embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class JinaProviderConfig(TypedDict, total=False):
    """Configuration for Jina provider."""

    api_key: str
    model_name: Annotated[str, "jina-embeddings-v2-base-en"]


class JinaProviderSpec(TypedDict, total=False):
    """Jina provider specification."""

    provider: Required[Literal["jina"]]
    config: JinaProviderConfig

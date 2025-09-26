"""Type definitions for SentenceTransformer embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class SentenceTransformerProviderConfig(TypedDict, total=False):
    """Configuration for SentenceTransformer provider."""

    model_name: Annotated[str, "all-MiniLM-L6-v2"]
    device: Annotated[str, "cpu"]
    normalize_embeddings: Annotated[bool, False]


class SentenceTransformerProviderSpec(TypedDict):
    """SentenceTransformer provider specification."""

    provider: Required[Literal["sentence-transformer"]]
    config: SentenceTransformerProviderConfig

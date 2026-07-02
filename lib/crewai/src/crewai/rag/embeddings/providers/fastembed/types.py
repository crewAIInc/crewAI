"""Type definitions for FastEmbed embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class FastEmbedProviderConfig(TypedDict, total=False):
    """Configuration for FastEmbed provider."""

    model_name: Annotated[str, "sentence-transformers/all-MiniLM-L6-v2"]
    cache_dir: str | None
    threads: int | None
    providers: list[str] | None
    cuda: Annotated[bool, False]
    device_ids: list[int] | None
    lazy_load: Annotated[bool, False]
    batch_size: Annotated[int, 256]
    parallel: int | None


class FastEmbedProviderSpec(TypedDict, total=False):
    """FastEmbed provider specification."""

    provider: Required[Literal["fastembed"]]
    config: FastEmbedProviderConfig

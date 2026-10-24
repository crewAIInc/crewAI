"""Type definitions for DeepInfra embedding providers."""

from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


class DeepInfraProviderConfig(TypedDict, total=False):
    """Configuration for DeepInfra provider."""

    api_key: str
    model: str
    model_name: Annotated[str, "Qwen/Qwen3-Embedding-8B"]
    api_base: str
    default_headers: dict[str, Any] | None


class DeepInfraProviderSpec(TypedDict, total=False):
    """DeepInfra provider specification."""

    provider: Required[Literal["deepinfra"]]
    config: DeepInfraProviderConfig

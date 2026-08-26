"""Type definitions for OpenRouter embedding providers."""

from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


class OpenRouterProviderConfig(TypedDict, total=False):
    """Configuration for OpenRouter provider."""

    api_key: str
    model_name: Annotated[str, "openai/text-embedding-3-small"]
    api_base: str
    default_headers: dict[str, Any]
    dimensions: int
    organization_id: str


class OpenRouterProviderSpec(TypedDict, total=False):
    """OpenRouter provider specification."""

    provider: Required[Literal["openrouter"]]
    config: OpenRouterProviderConfig

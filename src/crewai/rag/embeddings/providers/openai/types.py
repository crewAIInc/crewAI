"""Type definitions for OpenAI embedding providers."""

from typing import Annotated, Any, Literal, Required, TypedDict


class OpenAIProviderConfig(TypedDict, total=False):
    """Configuration for OpenAI provider."""

    api_key: Required[str]
    model_name: Annotated[str, "text-embedding-ada-002"]
    api_base: str
    api_type: str
    api_version: str
    default_headers: dict[str, Any]
    dimensions: int
    deployment_id: str
    organization_id: str


class OpenAIProviderSpec(TypedDict):
    """OpenAI provider specification."""

    provider: Literal["openai"]
    config: OpenAIProviderConfig

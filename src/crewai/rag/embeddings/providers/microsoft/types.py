"""Type definitions for Microsoft Azure embedding providers."""

from typing import Annotated, Any, Literal, Required, TypedDict


class AzureProviderConfig(TypedDict, total=False):
    """Configuration for Azure provider."""

    api_key: Required[str]
    api_base: str
    api_type: Annotated[str, "azure"]
    api_version: str
    model_name: Annotated[str, "text-embedding-ada-002"]
    default_headers: dict[str, Any]
    dimensions: int
    deployment_id: str
    organization_id: str


class AzureProviderSpec(TypedDict):
    """Azure provider specification."""

    provider: Literal["azure"]
    config: AzureProviderConfig

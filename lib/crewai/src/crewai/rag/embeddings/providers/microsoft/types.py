"""Type definitions for Microsoft Azure embedding providers."""

from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


class AzureProviderConfig(TypedDict, total=False):
    """Configuration for Azure provider."""

    api_key: str
    api_base: str
    api_type: Annotated[str, "azure"]
    api_version: str
    model_name: Annotated[str, "text-embedding-ada-002"]
    default_headers: dict[str, Any]
    dimensions: int
    deployment_id: Required[str]
    organization_id: str


class AzureProviderSpec(TypedDict, total=False):
    """Azure provider specification."""

    provider: Required[Literal["azure"]]
    config: AzureProviderConfig

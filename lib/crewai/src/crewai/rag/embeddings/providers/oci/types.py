"""Type definitions for OCI embedding providers."""

from typing import Annotated, Any, Literal

from typing_extensions import Required, TypedDict


class OCIProviderConfig(TypedDict, total=False):
    """Configuration for OCI embedding provider."""

    model_name: Annotated[str, "cohere.embed-english-v3.0"]
    compartment_id: str
    service_endpoint: str
    region: str
    auth_type: str
    auth_profile: str
    auth_file_location: str
    truncate: str
    input_type: str
    output_dimensions: int
    batch_size: int
    timeout: tuple[int, int]
    client: Any


class OCIProviderSpec(TypedDict, total=False):
    """OCI provider specification."""

    provider: Required[Literal["oci"]]
    config: OCIProviderConfig

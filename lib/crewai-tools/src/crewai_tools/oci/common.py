from __future__ import annotations

import os
from typing import Any

from crewai.utilities.oci import (
    create_oci_client_kwargs as shared_create_oci_client_kwargs,
    get_oci_module as shared_get_oci_module,
)


DEFAULT_OCI_REGION = "us-chicago-1"


def get_oci_module() -> Any:
    """Import the OCI SDK lazily so optional dependencies stay optional."""
    try:
        return shared_get_oci_module()
    except ImportError:
        raise ImportError(
            "`oci` package not found, please install the optional dependency with "
            "`uv add 'crewai-tools[oci]'`"
        ) from None


def create_oci_client_kwargs(
    *,
    auth_type: str,
    auth_profile: str,
    auth_file_location: str,
    service_endpoint: str | None = None,
    timeout: tuple[int, int] = (10, 120),
) -> dict[str, Any]:
    """Build standard OCI SDK client kwargs shared by the tool integrations."""
    return shared_create_oci_client_kwargs(
        auth_type=auth_type,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
        service_endpoint=service_endpoint,
        timeout=timeout,
        oci_module=get_oci_module(),
    )


def parse_object_storage_path(file_path: str) -> tuple[str | None, str, str]:
    """Parse an OCI Object Storage path.

    Supported formats:
    - `oci://bucket/path/to/object.txt`
    - `oci://namespace@bucket/path/to/object.txt`
    """
    normalized = file_path.removeprefix("oci://")
    bucket_part, _, object_name = normalized.partition("/")
    if not bucket_part or not object_name:
        raise ValueError(
            "OCI Object Storage paths must be in the form "
            "`oci://bucket/path` or `oci://namespace@bucket/path`."
        )

    if "@" in bucket_part:
        namespace_name, bucket_name = bucket_part.split("@", 1)
        return namespace_name, bucket_name, object_name

    return None, bucket_part, object_name


def get_region() -> str:
    """Return the default OCI region for tools that support region fallbacks."""
    return os.getenv("OCI_REGION", DEFAULT_OCI_REGION)

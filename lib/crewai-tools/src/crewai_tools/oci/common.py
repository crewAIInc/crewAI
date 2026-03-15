from __future__ import annotations

import os
from typing import Any


DEFAULT_OCI_REGION = "us-chicago-1"


def get_oci_module() -> Any:
    """Import the OCI SDK lazily so optional dependencies stay optional."""
    try:
        import oci  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "`oci` package not found, please install the optional dependency with "
            "`uv add 'crewai-tools[oci]'`"
        ) from None
    return oci


def create_oci_client_kwargs(
    *,
    auth_type: str,
    auth_profile: str,
    auth_file_location: str,
    service_endpoint: str | None = None,
    timeout: tuple[int, int] = (10, 120),
) -> dict[str, Any]:
    """Build standard OCI SDK client kwargs shared by the tool integrations."""
    oci = get_oci_module()
    client_kwargs: dict[str, Any] = {
        "config": {},
        "service_endpoint": service_endpoint,
        "retry_strategy": oci.retry.DEFAULT_RETRY_STRATEGY,
        "timeout": timeout,
    }

    auth_type_upper = auth_type.upper()
    if auth_type_upper == "API_KEY":
        client_kwargs["config"] = oci.config.from_file(
            file_location=auth_file_location,
            profile_name=auth_profile,
        )
    elif auth_type_upper == "SECURITY_TOKEN":
        config = oci.config.from_file(
            file_location=auth_file_location,
            profile_name=auth_profile,
        )
        private_key = oci.signer.load_private_key_from_file(config["key_file"], None)
        with open(config["security_token_file"], encoding="utf-8") as file:
            security_token = file.read()
        client_kwargs["config"] = config
        client_kwargs["signer"] = oci.auth.signers.SecurityTokenSigner(
            security_token, private_key
        )
    elif auth_type_upper == "INSTANCE_PRINCIPAL":
        client_kwargs["signer"] = (
            oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        )
    elif auth_type_upper == "RESOURCE_PRINCIPAL":
        client_kwargs["signer"] = oci.auth.signers.get_resource_principals_signer()
    else:
        valid_types = [
            "API_KEY",
            "SECURITY_TOKEN",
            "INSTANCE_PRINCIPAL",
            "RESOURCE_PRINCIPAL",
        ]
        raise ValueError(
            f"Invalid OCI auth_type '{auth_type}'. Valid values: {valid_types}"
        )

    return client_kwargs


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

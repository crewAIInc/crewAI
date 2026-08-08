from __future__ import annotations

from typing import Any


def get_oci_module() -> Any:
    """Import the OCI SDK lazily for optional CrewAI OCI integrations."""
    try:
        import oci  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            'OCI support is not available, to install: uv add "crewai[oci]"'
        ) from None
    return oci


def create_oci_client_kwargs(
    *,
    auth_type: str,
    service_endpoint: str | None,
    auth_file_location: str,
    auth_profile: str,
    timeout: tuple[int, int],
    oci_module: Any | None = None,
) -> dict[str, Any]:
    """Build OCI SDK client kwargs for the supported auth modes."""
    oci = oci_module or get_oci_module()
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
        key_file = config["key_file"]
        security_token_file = config["security_token_file"]
        private_key = oci.signer.load_private_key_from_file(key_file, None)
        with open(security_token_file, encoding="utf-8") as file:
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

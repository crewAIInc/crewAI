import os
import warnings


def platform_tls_verify() -> bool:
    """TLS verification setting for CrewAI platform API requests.

    Verification is ON by default and should stay on: these requests carry the
    platform integration token, so an unverified connection exposes it to
    man-in-the-middle attacks. It can be disabled only via an explicit opt-out
    (``CREWAI_PLATFORM_INSECURE_SKIP_TLS_VERIFY``; the legacy ``CREWAI_FACTORY``
    is still honored for internal builds), and doing so warns loudly. To trust a
    self-signed endpoint *securely*, leave verification on and point requests at
    the CA bundle via the standard ``REQUESTS_CA_BUNDLE`` environment variable.
    """
    if (
        os.getenv("CREWAI_PLATFORM_INSECURE_SKIP_TLS_VERIFY", "false").strip().lower()
        == "true"
        or os.getenv("CREWAI_FACTORY", "false").strip().lower() == "true"
    ):
        warnings.warn(
            "TLS certificate verification is DISABLED for CrewAI platform API "
            "requests; the integration token is sent over an unverified connection "
            "and is exposed to man-in-the-middle attacks. Unset "
            "CREWAI_PLATFORM_INSECURE_SKIP_TLS_VERIFY (and the legacy CREWAI_FACTORY). "
            "To trust a self-signed endpoint securely, keep verification on and set "
            "REQUESTS_CA_BUNDLE to your CA instead.",
            UserWarning,
            stacklevel=2,
        )
        return False
    return True


def get_platform_api_base_url() -> str:
    """Get the platform API base URL from environment or use default."""
    base_url = os.getenv("CREWAI_PLUS_URL", "https://app.crewai.com")
    return f"{base_url}/crewai_plus/api/v1/integrations"


def get_platform_integration_token() -> str:
    """Get the platform API base URL from environment or use default."""
    token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN") or ""
    if not token:
        raise ValueError(
            "No platform integration token found, please set the CREWAI_PLATFORM_INTEGRATION_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

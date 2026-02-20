import os

from crewai.context import get_platform_integration_token as _get_context_token


def get_platform_api_base_url() -> str:
    """Get the platform API base URL from environment or use default."""
    base_url = os.getenv("CREWAI_PLUS_URL", "https://app.crewai.com")
    return f"{base_url}/crewai_plus/api/v1/integrations"


def get_platform_integration_token() -> str:
    """Get the platform integration token from the context.
    Fallback to the environment variable if no token has been set in the context.

    Raises:
        ValueError: If no token has been set in the context.
    """
    token = _get_context_token() or os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")
    if not token:
        raise ValueError(
            "No platform integration token found. "
            "Set it via platform_integration_context() or set_platform_integration_token()."
        )
    return token

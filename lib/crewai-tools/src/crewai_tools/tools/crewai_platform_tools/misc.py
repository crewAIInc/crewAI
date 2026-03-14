import os

from crewai.context import get_platform_integration_token as _get_context_token


def get_platform_api_base_url() -> str:
    """Get the platform API base URL from environment or use default."""
    base_url = os.getenv("CREWAI_PLUS_URL", "https://app.crewai.com")
    return f"{base_url}/crewai_plus/api/v1/integrations"


def get_platform_integration_token() -> str | None:
    return _get_context_token() or os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")

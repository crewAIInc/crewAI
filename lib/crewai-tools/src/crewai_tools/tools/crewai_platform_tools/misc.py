import os


def get_platform_integration_token() -> str:
    """Get the platform API base URL from environment or use default."""
    token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN") or ""
    if not token:
        raise ValueError(
            "No platform integration token found, please set the CREWAI_PLATFORM_INTEGRATION_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

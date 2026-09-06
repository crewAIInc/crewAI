import os


def get_platform_integration_token() -> str:
    """Get the platform integration token from the environment."""
    token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN") or ""
    if not token:
        raise ValueError(
            "No platform integration token found, please set the CREWAI_PLATFORM_INTEGRATION_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

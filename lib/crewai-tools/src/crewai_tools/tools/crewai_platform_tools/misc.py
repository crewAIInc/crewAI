import os


def get_platform_integration_token() -> str:
    """Get the Platform Enterprise Action Auth Token from the environment."""
    token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN") or ""
    if not token:
        raise ValueError(
            "No Platform Enterprise Action Auth Token found, please set the "
            "CREWAI_PLATFORM_INTEGRATION_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

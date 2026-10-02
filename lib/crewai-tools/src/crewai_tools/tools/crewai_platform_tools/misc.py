import os


def get_platform_integration_token() -> str:
    """Get the Platform integration token from the environment."""
    token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "")
    if not token:
        raise ValueError(
            "No Platform integration token found, please set the "
            "CREWAI_PLATFORM_INTEGRATION_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

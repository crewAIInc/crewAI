import os


def get_platform_integration_token() -> str:
    """Get the Platform integration token from the environment."""
    token = os.getenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", "")
    if not token:
        raise ValueError(
            "No Enterprise Action Auth Token found, please set the "
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

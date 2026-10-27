import os


def get_platform_integration_token() -> str:
    """Get the Platform integration token from the environment."""
    token = os.getenv("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", "")
    # Projects created before #7990 still set the old name, which is no longer
    # read; say so instead of reporting the token as missing. Remove in 2.0.
    if not token and os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN"):
        raise ValueError(
            "CREWAI_PLATFORM_INTEGRATION_TOKEN is set, but CrewAI now reads the "
            "Enterprise Action Auth Token only from "
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN. Rename the variable in your "
            ".env; the value stays the same."
        )
    if not token:
        raise ValueError(
            "No Enterprise Action Auth Token found, please set the "
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN environment variable"
        )
    return token  # TODO: Use context manager to get token

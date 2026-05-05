"""Authentication token retrieval."""

from crewai_core.token_manager import TokenManager


class AuthError(Exception):
    """Raised when authentication fails."""


def get_auth_token() -> str:
    """Get the authentication token."""
    access_token = TokenManager().get_token()
    if not access_token:
        raise AuthError("No token found, make sure you are logged in")
    return access_token

"""Authentication token retrieval."""

from __future__ import annotations

from crewai_core.token_manager import TokenManager


class AuthError(Exception):
    """Raised when authentication fails."""


def get_auth_token() -> str:
    """Return the saved authentication token; raise ``AuthError`` if missing."""
    access_token = TokenManager().get_token()
    if not access_token:
        raise AuthError("No token found, make sure you are logged in")
    return access_token

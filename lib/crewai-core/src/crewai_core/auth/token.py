"""Authentication token retrieval."""

from __future__ import annotations

from crewai_core.auth.oauth2 import REFRESH_MARGIN_SECONDS, AuthenticationCommand
from crewai_core.token_manager import TokenManager


class AuthError(Exception):
    """Raised when authentication fails."""


def get_auth_token() -> str:
    """Return the saved authentication token; raise ``AuthError`` if missing.

    An access token about to expire is renewed with the saved refresh token first.
    """
    token_manager = TokenManager()
    access_token = token_manager.get_token(REFRESH_MARGIN_SECONDS)
    if access_token is None and token_manager.get_refresh_token() is not None:
        access_token = AuthenticationCommand().refresh_access_token()
    # A failed refresh still leaves a token inside the margin usable.
    access_token = access_token or token_manager.get_token()
    if not access_token:
        raise AuthError("No token found, make sure you are logged in")
    return access_token

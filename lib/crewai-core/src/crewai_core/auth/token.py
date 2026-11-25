"""Authentication token retrieval."""

from __future__ import annotations

import logging
import time

from crewai_core.auth.oauth2 import REFRESH_MARGIN_SECONDS, AuthenticationCommand
from crewai_core.token_manager import TokenManager


logger = logging.getLogger(__name__)

# After a failed refresh, later calls in this process (the next trace grant, the next
# PlusAPI client) go without one for a while instead of each waiting on the network
# again. In-process on purpose: the next command retries at once, the network may be back.
_REFRESH_RETRY_SECONDS = 300
_refresh_retry_at: dict[str, float] = {}


class AuthError(Exception):
    """Raised when authentication fails."""


def get_auth_token(*, refresh: bool = True) -> str:
    """Return the saved authentication token; raise ``AuthError`` if missing.

    Args:
        refresh: Renew an access token about to expire with the saved refresh
            token first, which may send one request to the login provider. Pass
            False where only the saved token is wanted and no request may be sent.

    Raises:
        AuthError: If there is no usable token, including when renewal failed.
    """
    token_manager = TokenManager()
    access_token: str | None = None
    if refresh:
        access_token = token_manager.get_token(REFRESH_MARGIN_SECONDS)
        if access_token is None and token_manager.get_refresh_token() is not None:
            access_token = _refresh(token_manager)
    # A skipped or failed refresh still leaves a token inside the margin usable.
    access_token = access_token or token_manager.get_token()
    if not access_token:
        raise AuthError("No token found, make sure you are logged in")
    return access_token


def _refresh(token_manager: TokenManager) -> str | None:
    store = token_manager.token_path
    if _refresh_retry_at.get(store, 0.0) > time.monotonic():
        return None
    try:
        access_token = AuthenticationCommand().refresh_access_token()
    # Renewal is best effort, and callers such as tracing handle only AuthError. The
    # settings, the lock backend (file, Redis or a custom one) and the provider each
    # fail with their own exception types; any of them means "no renewed token".
    except Exception:
        logger.debug("Could not renew the saved CrewAI login", exc_info=True)
        access_token = None
    if access_token is None:
        _refresh_retry_at[store] = time.monotonic() + _REFRESH_RETRY_SECONDS
    else:
        _refresh_retry_at.pop(store, None)
    return access_token

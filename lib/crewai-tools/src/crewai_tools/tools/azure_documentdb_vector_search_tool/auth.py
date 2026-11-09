from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pymongo.auth_oidc import OIDCCallback, OIDCCallbackContext, OIDCCallbackResult


if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential


class DocumentDBOIDCCallback(OIDCCallback):
    """Acquire renewable DocumentDB tokens using an Azure credential."""

    def __init__(self, credential: TokenCredential) -> None:
        """Retain the caller's Azure credential for token acquisition."""
        self._credential = credential

    def fetch(self, context: OIDCCallbackContext) -> OIDCCallbackResult:
        """Return a DocumentDB access token and its remaining lifetime."""
        token = self._credential.get_token(
            "https://ossrdbms-aad.database.windows.net/.default"
        )
        return OIDCCallbackResult(
            access_token=token.token,
            expires_in_seconds=max(0, token.expires_on - time.time()),
        )

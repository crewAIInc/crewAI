"""Device flow must use the RFC 8628 default poll interval when omitted."""

from unittest.mock import MagicMock, patch

import pytest

from crewai_core.auth.oauth2 import AuthenticationCommand


def test_missing_interval_defaults_to_five_seconds():
    oauth = object.__new__(AuthenticationCommand)
    oauth.oauth2_provider = MagicMock()
    oauth.oauth2_provider.get_token_url.return_value = "https://example.test/token"
    oauth.oauth2_provider.get_client_id.return_value = "client"

    with (
        patch("crewai_core.auth.oauth2.httpx.post") as post,
        patch("crewai_core.auth.oauth2.time") as clock,
        patch("crewai_core.auth.oauth2.console"),
    ):
        post.return_value.json.return_value = {"error": "authorization_pending"}
        clock.monotonic.side_effect = [0, 0, 601]

        with pytest.raises(SystemExit):
            oauth._poll_for_token({"device_code": "dc", "expires_in": 600})

        clock.sleep.assert_called_with(5)

"""Unit tests for A2A authentication schemes."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

from crewai.experimental.a2a import (
    APIKeyAuth,
    BearerTokenAuth,
    HTTPBasicAuth,
    HTTPDigestAuth,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
    create_auth_from_agent_card,
)
import httpx
import pytest


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    """Provide an async HTTP client for tests."""
    async with httpx.AsyncClient() as client:
        yield client


class TestBearerTokenAuth:
    """Tests for BearerTokenAuth."""

    @pytest.mark.asyncio
    async def test_apply_auth_adds_bearer_token(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that Bearer token is added to Authorization header."""
        auth = BearerTokenAuth(token="test-token-123")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        assert result["Authorization"] == "Bearer test-token-123"

    @pytest.mark.asyncio
    async def test_apply_auth_overwrites_existing_auth(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that Bearer token overwrites existing Authorization header."""
        auth = BearerTokenAuth(token="new-token")
        headers = {"Authorization": "Bearer old-token"}

        result = await auth.apply_auth(http_client, headers)

        assert result["Authorization"] == "Bearer new-token"

    def test_configure_client_does_nothing(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that configure_client is a no-op for Bearer tokens."""
        auth = BearerTokenAuth(token="test-token")
        auth.configure_client(http_client)
        # Should not raise any errors


class TestHTTPBasicAuth:
    """Tests for HTTPBasicAuth."""

    @pytest.mark.asyncio
    async def test_apply_auth_adds_basic_credentials(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that Basic auth credentials are encoded and added."""
        auth = HTTPBasicAuth(username="user", password="pass")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        # Verify the header format
        assert "Authorization" in result
        assert result["Authorization"].startswith("Basic ")

        # Decode and verify credentials
        encoded = result["Authorization"].split(" ")[1]
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "user:pass"

    @pytest.mark.asyncio
    async def test_apply_auth_with_special_characters(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test Basic auth with special characters in credentials."""
        auth = HTTPBasicAuth(username="user@example.com", password="p@ss:w0rd!")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        encoded = result["Authorization"].split(" ")[1]
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "user@example.com:p@ss:w0rd!"

    def test_configure_client_does_nothing(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that configure_client is a no-op for Basic auth."""
        auth = HTTPBasicAuth(username="user", password="pass")
        auth.configure_client(http_client)


class TestHTTPDigestAuth:
    """Tests for HTTPDigestAuth."""

    @pytest.mark.asyncio
    async def test_apply_auth_returns_headers_unchanged(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that apply_auth doesn't modify headers (digest is handled by httpx)."""
        auth = HTTPDigestAuth(username="user", password="pass")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        assert result == headers
        assert "Authorization" not in result

    def test_configure_client_requires_httpx_auth(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that configure_client requires httpx-auth library."""
        auth = HTTPDigestAuth(username="user", password="pass")

        # This will fail if httpx-auth is not installed
        with pytest.raises(ImportError, match="httpx-auth required"):
            auth.configure_client(http_client)


class TestAPIKeyAuth:
    """Tests for APIKeyAuth."""

    @pytest.mark.asyncio
    async def test_apply_auth_header_location(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test API key in header location."""
        auth = APIKeyAuth(api_key="key-123", location="header", name="X-API-Key")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        assert result["X-API-Key"] == "key-123"

    @pytest.mark.asyncio
    async def test_apply_auth_cookie_location(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test API key in cookie location."""
        auth = APIKeyAuth(api_key="key-456", location="cookie", name="api_token")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        assert result["Cookie"] == "api_token=key-456"

    @pytest.mark.asyncio
    async def test_apply_auth_query_location(
        self, http_client: httpx.AsyncClient
    ) -> None:
        """Test that query location doesn't modify headers."""
        auth = APIKeyAuth(api_key="key-789", location="query", name="api_key")
        headers: dict[str, str] = {}

        result = await auth.apply_auth(http_client, headers)

        # Query params are handled in configure_client, not apply_auth
        assert result == headers

    def test_configure_client_query_adds_event_hook(self) -> None:
        """Test that configure_client adds event hook for query params."""
        auth = APIKeyAuth(api_key="key-xyz", location="query", name="api_key")

        # Create a real client for this test
        client = httpx.AsyncClient()
        initial_hooks = len(client.event_hooks.get("request", []))

        auth.configure_client(client)

        # Should have added one hook
        assert len(client.event_hooks["request"]) == initial_hooks + 1


class TestOAuth2ClientCredentials:
    """Tests for OAuth2ClientCredentials."""

    @pytest.mark.asyncio
    async def test_fetch_token_and_apply_auth(self) -> None:
        """Test fetching OAuth2 token and applying it."""
        auth = OAuth2ClientCredentials(
            token_url="https://auth.example.com/token",
            client_id="client-123",
            client_secret="secret-456",
            scopes=["read", "write"],
        )

        # Mock the HTTP client response
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "access-token-xyz",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        headers: dict[str, str] = {}
        result = await auth.apply_auth(mock_client, headers)

        # Verify token was fetched
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://auth.example.com/token"
        assert call_args[1]["data"]["grant_type"] == "client_credentials"
        assert call_args[1]["data"]["client_id"] == "client-123"
        assert call_args[1]["data"]["client_secret"] == "secret-456"
        assert call_args[1]["data"]["scope"] == "read write"

        # Verify token was applied
        assert result["Authorization"] == "Bearer access-token-xyz"

    @pytest.mark.asyncio
    async def test_token_caching(self) -> None:
        """Test that token is cached and not re-fetched unnecessarily."""
        auth = OAuth2ClientCredentials(
            token_url="https://auth.example.com/token",
            client_id="client-123",
            client_secret="secret-456",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "token-abc",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        # First call fetches token
        await auth.apply_auth(mock_client, {})
        assert mock_client.post.call_count == 1

        # Second call uses cached token
        await auth.apply_auth(mock_client, {})
        assert mock_client.post.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_token_refresh_on_expiry(self) -> None:
        """Test that token is refreshed when expired."""
        auth = OAuth2ClientCredentials(
            token_url="https://auth.example.com/token",
            client_id="client-123",
            client_secret="secret-456",
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "token-def",
            "expires_in": -1,  # Already expired
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        # First call fetches token
        await auth.apply_auth(mock_client, {})
        assert mock_client.post.call_count == 1

        # Second call refreshes expired token
        await auth.apply_auth(mock_client, {})
        assert mock_client.post.call_count == 2


class TestOAuth2AuthorizationCode:
    """Tests for OAuth2AuthorizationCode."""

    @pytest.mark.asyncio
    async def test_requires_callback_to_be_set(self) -> None:
        """Test that authorization callback must be set before use."""
        auth = OAuth2AuthorizationCode(
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            client_id="client-123",
            client_secret="secret-456",
            redirect_uri="https://app.example.com/callback",
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        headers: dict[str, str] = {}

        # Should raise error when callback is not set
        with pytest.raises(ValueError, match="Authorization callback not set"):
            await auth.apply_auth(mock_client, headers)

    @pytest.mark.asyncio
    async def test_authorization_flow(self) -> None:
        """Test full authorization code flow."""
        auth = OAuth2AuthorizationCode(
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            client_id="client-123",
            client_secret="secret-456",
            redirect_uri="https://app.example.com/callback",
            scopes=["openid", "profile"],
        )

        # Mock the callback
        async def mock_callback(url: str) -> str:
            assert "response_type=code" in url
            assert "client_id=client-123" in url
            return "auth-code-xyz"

        auth.set_authorization_callback(mock_callback)

        # Mock token response
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "access-token-abc",
            "refresh_token": "refresh-token-def",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        headers: dict[str, str] = {}
        result = await auth.apply_auth(mock_client, headers)

        # Verify token exchange
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[1]["data"]["grant_type"] == "authorization_code"
        assert call_args[1]["data"]["code"] == "auth-code-xyz"

        # Verify token was applied
        assert result["Authorization"] == "Bearer access-token-abc"

    @pytest.mark.asyncio
    async def test_token_refresh(self) -> None:
        """Test token refresh with refresh token."""
        import time

        auth = OAuth2AuthorizationCode(
            authorization_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            client_id="client-123",
            client_secret="secret-456",
            redirect_uri="https://app.example.com/callback",
        )

        # Set initial tokens manually and make sure they're expired
        auth._access_token = "old-token"
        auth._refresh_token = "refresh-token-xyz"
        auth._token_expires_at = time.time() - 100  # Expired 100 seconds ago

        # Mock refresh response
        mock_response = Mock()
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        headers: dict[str, str] = {}
        result = await auth.apply_auth(mock_client, headers)

        # Verify refresh token was used
        assert mock_client.post.called
        call_args = mock_client.post.call_args
        assert call_args is not None
        assert call_args.kwargs["data"]["grant_type"] == "refresh_token"
        assert call_args.kwargs["data"]["refresh_token"] == "refresh-token-xyz"

        # Verify new token was applied
        assert result["Authorization"] == "Bearer new-access-token"


class TestCreateAuthFromAgentCard:
    """Tests for create_auth_from_agent_card helper."""

    def test_no_security_returns_none(self) -> None:
        """Test that no auth is created when AgentCard has no security."""
        mock_card = Mock()
        mock_card.security = None
        mock_card.security_schemes = None

        result = create_auth_from_agent_card(mock_card, {})

        assert result is None

    def test_bearer_token_from_agent_card(self) -> None:
        """Test creating Bearer auth from AgentCard."""
        mock_card = Mock()
        mock_card.security = [{"bearerAuth": []}]
        mock_card.security_schemes = {
            "bearerAuth": {"type": "http", "scheme": "bearer"}
        }

        result = create_auth_from_agent_card(mock_card, {"token": "my-token"})

        assert isinstance(result, BearerTokenAuth)
        assert result.token == "my-token"

    def test_basic_auth_from_agent_card(self) -> None:
        """Test creating Basic auth from AgentCard."""
        mock_card = Mock()
        mock_card.security = [{"basicAuth": []}]
        mock_card.security_schemes = {"basicAuth": {"type": "http", "scheme": "basic"}}

        result = create_auth_from_agent_card(
            mock_card, {"username": "user", "password": "pass"}
        )

        assert isinstance(result, HTTPBasicAuth)
        assert result.username == "user"
        assert result.password == "pass"

    def test_api_key_auth_from_agent_card(self) -> None:
        """Test creating API Key auth from AgentCard."""
        mock_card = Mock()
        mock_card.security = [{"apiKey": []}]
        mock_card.security_schemes = {
            "apiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        }

        result = create_auth_from_agent_card(mock_card, {"api_key": "key-123"})

        assert isinstance(result, APIKeyAuth)
        assert result.api_key == "key-123"
        assert result.location == "header"
        assert result.name == "X-API-Key"

    def test_oauth2_client_credentials_from_agent_card(self) -> None:
        """Test creating OAuth2 Client Credentials from AgentCard."""
        mock_card = Mock()
        mock_card.security = [{"oauth2": ["read", "write"]}]
        mock_card.security_schemes = {
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "clientCredentials": {
                        "tokenUrl": "https://auth.example.com/token",
                        "scopes": {"read": "Read access", "write": "Write access"},
                    }
                },
            }
        }

        result = create_auth_from_agent_card(
            mock_card, {"client_id": "client-id", "client_secret": "client-secret"}
        )

        assert isinstance(result, OAuth2ClientCredentials)
        assert result.token_url == "https://auth.example.com/token"
        assert result.client_id == "client-id"
        assert result.client_secret == "client-secret"
        assert set(result.scopes) == {"read", "write"}

    def test_missing_scheme_returns_none(self) -> None:
        """Test that missing security scheme returns None."""
        mock_card = Mock()
        mock_card.security = [{"unknownAuth": []}]
        mock_card.security_schemes = {}

        result = create_auth_from_agent_card(mock_card, {})

        assert result is None

"""Authentication schemes for A2A protocol clients.

Supported authentication methods:
- Bearer tokens
- OAuth2 (Client Credentials, Authorization Code)
- API Keys (header, query, cookie)
- HTTP Basic authentication
- HTTP Digest authentication
- mTLS (mutual TLS) client certificate authentication
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import base64
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
import ssl
import time
from typing import TYPE_CHECKING, ClassVar, Literal
import urllib.parse

import httpx
from httpx import DigestAuth
from pydantic import BaseModel, ConfigDict, Field, FilePath, PrivateAttr
from typing_extensions import deprecated


if TYPE_CHECKING:
    import grpc  # type: ignore[import-untyped]


class TLSConfig(BaseModel):
    """TLS/mTLS configuration for secure client connections.

    Supports mutual TLS (mTLS) where the client presents a certificate to the server,
    and standard TLS with custom CA verification.

    Attributes:
        client_cert_path: Path to client certificate file (PEM format) for mTLS.
        client_key_path: Path to client private key file (PEM format) for mTLS.
        ca_cert_path: Path to CA certificate bundle for server verification.
        verify: Whether to verify server certificates. Set False only for development.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    client_cert_path: FilePath | None = Field(
        default=None,
        description="Path to client certificate file (PEM format) for mTLS",
    )
    client_key_path: FilePath | None = Field(
        default=None,
        description="Path to client private key file (PEM format) for mTLS",
    )
    ca_cert_path: FilePath | None = Field(
        default=None,
        description="Path to CA certificate bundle for server verification",
    )
    verify: bool = Field(
        default=True,
        description="Whether to verify server certificates. Set False only for development.",
    )

    def get_httpx_ssl_context(self) -> ssl.SSLContext | bool | str:
        """Build SSL context for httpx client.

        Returns:
            SSL context if certificates configured, True for default verification,
            False if verification disabled, or path to CA bundle.
        """
        if not self.verify:
            return False

        if self.client_cert_path and self.client_key_path:
            context = ssl.create_default_context()

            if self.ca_cert_path:
                context.load_verify_locations(cafile=str(self.ca_cert_path))

            context.load_cert_chain(
                certfile=str(self.client_cert_path),
                keyfile=str(self.client_key_path),
            )
            return context

        if self.ca_cert_path:
            return str(self.ca_cert_path)

        return True

    def get_grpc_credentials(self) -> grpc.ChannelCredentials | None:  # type: ignore[no-any-unimported]
        """Build gRPC channel credentials for secure connections.

        Returns:
            gRPC SSL credentials if certificates configured, None otherwise.
        """
        try:
            import grpc
        except ImportError:
            return None

        if not self.verify and not self.client_cert_path:
            return None

        root_certs: bytes | None = None
        private_key: bytes | None = None
        certificate_chain: bytes | None = None

        if self.ca_cert_path:
            root_certs = Path(self.ca_cert_path).read_bytes()

        if self.client_cert_path and self.client_key_path:
            private_key = Path(self.client_key_path).read_bytes()
            certificate_chain = Path(self.client_cert_path).read_bytes()

        return grpc.ssl_channel_credentials(
            root_certificates=root_certs,
            private_key=private_key,
            certificate_chain=certificate_chain,
        )


class ClientAuthScheme(ABC, BaseModel):
    """Base class for client-side authentication schemes.

    Client auth schemes apply credentials to outgoing requests.

    Attributes:
        tls: Optional TLS/mTLS configuration for secure connections.
    """

    tls: TLSConfig | None = Field(
        default=None,
        description="TLS/mTLS configuration for secure connections",
    )

    @abstractmethod
    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply authentication to request headers.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with authentication applied.
        """
        ...


@deprecated("Use ClientAuthScheme instead", category=FutureWarning)
class AuthScheme(ClientAuthScheme):
    """Deprecated: Use ClientAuthScheme instead."""


class BearerTokenAuth(ClientAuthScheme):
    """Bearer token authentication (Authorization: Bearer <token>).

    Attributes:
        token: Bearer token for authentication.
    """

    token: str = Field(description="Bearer token")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply Bearer token to Authorization header.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with Bearer token in Authorization header.
        """
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


class HTTPBasicAuth(ClientAuthScheme):
    """HTTP Basic authentication.

    Attributes:
        username: Username for Basic authentication.
        password: Password for Basic authentication.
    """

    username: str = Field(description="Username")
    password: str = Field(description="Password")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply HTTP Basic authentication.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with Basic auth in Authorization header.
        """
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
        return headers


class HTTPDigestAuth(ClientAuthScheme):
    """HTTP Digest authentication.

    Note: Uses httpx-auth library for digest implementation.

    Attributes:
        username: Username for Digest authentication.
        password: Password for Digest authentication.
    """

    username: str = Field(description="Username")
    password: str = Field(description="Password")

    _configured_client_id: int | None = PrivateAttr(default=None)

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Digest auth is handled by httpx auth flow, not headers.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Unchanged headers (Digest auth handled by httpx auth flow).
        """
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure client with Digest auth.

        Idempotent: Only configures the client once. Subsequent calls on the same
        client instance are no-ops to prevent overwriting auth configuration.

        Args:
            client: HTTP client to configure with Digest authentication.
        """
        client_id = id(client)
        if self._configured_client_id == client_id:
            return

        client.auth = DigestAuth(self.username, self.password)
        self._configured_client_id = client_id


class APIKeyAuth(ClientAuthScheme):
    """API Key authentication (header, query, or cookie).

    Attributes:
        api_key: API key value for authentication.
        location: Where to send the API key (header, query, or cookie).
        name: Parameter name for the API key (default: X-API-Key).
    """

    api_key: str = Field(description="API key value")
    location: Literal["header", "query", "cookie"] = Field(
        default="header", description="Where to send the API key"
    )
    name: str = Field(default="X-API-Key", description="Parameter name for the API key")

    _configured_client_ids: set[int] = PrivateAttr(default_factory=set)

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply API key authentication.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with API key (for header/cookie locations).
        """
        if self.location == "header":
            headers[self.name] = self.api_key
        elif self.location == "cookie":
            headers["Cookie"] = f"{self.name}={self.api_key}"
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure client for query param API keys.

        Idempotent: Only adds the request hook once per client instance.
        Subsequent calls on the same client are no-ops to prevent hook accumulation.

        Args:
            client: HTTP client to configure with query param API key hook.
        """
        if self.location == "query":
            client_id = id(client)
            if client_id in self._configured_client_ids:
                return

            async def _add_api_key_param(request: httpx.Request) -> None:
                url = httpx.URL(request.url)
                request.url = url.copy_add_param(self.name, self.api_key)

            client.event_hooks["request"].append(_add_api_key_param)
            self._configured_client_ids.add(client_id)


class OAuth2ClientCredentials(ClientAuthScheme):
    """OAuth2 Client Credentials flow authentication.

    Thread-safe implementation with asyncio.Lock to prevent concurrent token fetches
    when multiple requests share the same auth instance.

    Attributes:
        token_url: OAuth2 token endpoint URL.
        client_id: OAuth2 client identifier.
        client_secret: OAuth2 client secret.
        scopes: List of required OAuth2 scopes.
    """

    token_url: str = Field(description="OAuth2 token endpoint")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )

    _access_token: str | None = PrivateAttr(default=None)
    _token_expires_at: float | None = PrivateAttr(default=None)
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply OAuth2 access token to Authorization header.

        Uses asyncio.Lock to ensure only one coroutine fetches tokens at a time,
        preventing race conditions when multiple concurrent requests use the same
        auth instance.

        Args:
            client: HTTP client for making token requests.
            headers: Current request headers.

        Returns:
            Updated headers with OAuth2 access token in Authorization header.
        """
        if (
            self._access_token is None
            or self._token_expires_at is None
            or time.time() >= self._token_expires_at
        ):
            async with self._lock:
                if (
                    self._access_token is None
                    or self._token_expires_at is None
                    or time.time() >= self._token_expires_at
                ):
                    await self._fetch_token(client)

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    async def _fetch_token(self, client: httpx.AsyncClient) -> None:
        """Fetch OAuth2 access token using client credentials flow.

        Args:
            client: HTTP client for making token request.

        Raises:
            httpx.HTTPStatusError: If token request fails.
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if self.scopes:
            data["scope"] = " ".join(self.scopes)

        response = await client.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60


class OAuth2AuthorizationCode(ClientAuthScheme):
    """OAuth2 Authorization Code flow authentication.

    Thread-safe implementation with asyncio.Lock to prevent concurrent token operations.

    Note: Requires interactive authorization.

    Attributes:
        authorization_url: OAuth2 authorization endpoint URL.
        token_url: OAuth2 token endpoint URL.
        client_id: OAuth2 client identifier.
        client_secret: OAuth2 client secret.
        redirect_uri: OAuth2 redirect URI for callback.
        scopes: List of required OAuth2 scopes.
    """

    authorization_url: str = Field(description="OAuth2 authorization endpoint")
    token_url: str = Field(description="OAuth2 token endpoint")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    redirect_uri: str = Field(description="OAuth2 redirect URI")
    scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )

    _access_token: str | None = PrivateAttr(default=None)
    _refresh_token: str | None = PrivateAttr(default=None)
    _token_expires_at: float | None = PrivateAttr(default=None)
    _authorization_callback: Callable[[str], Awaitable[str]] | None = PrivateAttr(
        default=None
    )
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def set_authorization_callback(
        self, callback: Callable[[str], Awaitable[str]] | None
    ) -> None:
        """Set callback to handle authorization URL.

        Args:
            callback: Async function that receives authorization URL and returns auth code.
        """
        self._authorization_callback = callback

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply OAuth2 access token to Authorization header.

        Uses asyncio.Lock to ensure only one coroutine handles token operations
        (initial fetch or refresh) at a time.

        Args:
            client: HTTP client for making token requests.
            headers: Current request headers.

        Returns:
            Updated headers with OAuth2 access token in Authorization header.

        Raises:
            ValueError: If authorization callback is not set.
        """
        if self._access_token is None:
            if self._authorization_callback is None:
                msg = "Authorization callback not set. Use set_authorization_callback()"
                raise ValueError(msg)
            async with self._lock:
                if self._access_token is None:
                    await self._fetch_initial_token(client)
        elif self._token_expires_at and time.time() >= self._token_expires_at:
            async with self._lock:
                if self._token_expires_at and time.time() >= self._token_expires_at:
                    await self._refresh_access_token(client)

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    async def _fetch_initial_token(self, client: httpx.AsyncClient) -> None:
        """Fetch initial access token using authorization code flow.

        Args:
            client: HTTP client for making token request.

        Raises:
            ValueError: If authorization callback is not set.
            httpx.HTTPStatusError: If token request fails.
        """
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
        }
        auth_url = f"{self.authorization_url}?{urllib.parse.urlencode(params)}"

        if self._authorization_callback is None:
            msg = "Authorization callback not set"
            raise ValueError(msg)
        auth_code = await self._authorization_callback(auth_url)

        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }

        response = await client.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")

        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60

    async def _refresh_access_token(self, client: httpx.AsyncClient) -> None:
        """Refresh the access token using refresh token.

        Args:
            client: HTTP client for making token request.

        Raises:
            httpx.HTTPStatusError: If token refresh request fails.
        """
        if not self._refresh_token:
            await self._fetch_initial_token(client)
            return

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = await client.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        if "refresh_token" in token_data:
            self._refresh_token = token_data["refresh_token"]

        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60

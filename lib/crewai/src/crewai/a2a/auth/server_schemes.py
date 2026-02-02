"""Server-side authentication schemes for A2A protocol.

These schemes validate incoming requests to A2A server endpoints.

Supported authentication methods:
- Simple token validation with static bearer tokens
- OpenID Connect with JWT validation using JWKS
- OAuth2 with JWT validation or token introspection
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import os
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import jwt
from jwt import PyJWKClient
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    PrivateAttr,
    SecretStr,
    model_validator,
)
from typing_extensions import Self


if TYPE_CHECKING:
    from a2a.types import OAuth2SecurityScheme


logger = logging.getLogger(__name__)


try:
    from fastapi import HTTPException, status as http_status

    HTTP_401_UNAUTHORIZED = http_status.HTTP_401_UNAUTHORIZED
    HTTP_500_INTERNAL_SERVER_ERROR = http_status.HTTP_500_INTERNAL_SERVER_ERROR
    HTTP_503_SERVICE_UNAVAILABLE = http_status.HTTP_503_SERVICE_UNAVAILABLE
except ImportError:

    class HTTPException(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Fallback HTTPException when FastAPI is not installed."""

        def __init__(
            self,
            status_code: int,
            detail: str | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.status_code = status_code
            self.detail = detail
            self.headers = headers
            super().__init__(detail)

    HTTP_401_UNAUTHORIZED = 401
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_503_SERVICE_UNAVAILABLE = 503


def _coerce_secret_str(v: str | SecretStr | None) -> SecretStr | None:
    """Coerce string to SecretStr."""
    if v is None or isinstance(v, SecretStr):
        return v
    return SecretStr(v)


CoercedSecretStr = Annotated[SecretStr, BeforeValidator(_coerce_secret_str)]

JWTAlgorithm = Literal[
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
]


@dataclass
class AuthenticatedUser:
    """Result of successful authentication.

    Attributes:
        token: The original token that was validated.
        scheme: Name of the authentication scheme used.
        claims: JWT claims from OIDC or OAuth2 authentication.
    """

    token: str
    scheme: str
    claims: dict[str, Any] | None = None


class ServerAuthScheme(ABC, BaseModel):
    """Base class for server-side authentication schemes.

    Each scheme validates incoming requests and returns an AuthenticatedUser
    on success, or raises HTTPException on failure.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @abstractmethod
    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Authenticate the provided token.

        Args:
            token: The bearer token to authenticate.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If authentication fails.
        """
        ...


class SimpleTokenAuth(ServerAuthScheme):
    """Simple bearer token authentication.

    Validates tokens against a configured static token or AUTH_TOKEN env var.

    Attributes:
        token: Expected token value. Falls back to AUTH_TOKEN env var if not set.
    """

    token: CoercedSecretStr | None = Field(
        default=None,
        description="Expected token. Falls back to AUTH_TOKEN env var.",
    )

    def _get_expected_token(self) -> str | None:
        """Get the expected token value."""
        if self.token:
            return self.token.get_secret_value()
        return os.environ.get("AUTH_TOKEN")

    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Authenticate using simple token comparison.

        Args:
            token: The bearer token to authenticate.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If authentication fails.
        """
        expected = self._get_expected_token()

        if expected is None:
            logger.warning(
                "Simple token authentication failed",
                extra={"reason": "no_token_configured"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Authentication not configured",
            )

        if token != expected:
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials",
            )

        return AuthenticatedUser(
            token=token,
            scheme="simple_token",
        )


class OIDCAuth(ServerAuthScheme):
    """OpenID Connect authentication.

    Validates JWTs using JWKS with caching support via PyJWT.

    Attributes:
        issuer: The OpenID Connect issuer URL.
        audience: The expected audience claim.
        jwks_url: Optional explicit JWKS URL. Derived from issuer if not set.
        algorithms: List of allowed signing algorithms.
        required_claims: List of claims that must be present in the token.
        jwks_cache_ttl: TTL for JWKS cache in seconds.
        clock_skew_seconds: Allowed clock skew for token validation.
    """

    issuer: HttpUrl = Field(
        description="OpenID Connect issuer URL (e.g., https://auth.example.com)"
    )
    audience: str = Field(description="Expected audience claim (e.g., api://my-agent)")
    jwks_url: HttpUrl | None = Field(
        default=None,
        description="Explicit JWKS URL. Derived from issuer if not set.",
    )
    algorithms: list[str] = Field(
        default_factory=lambda: ["RS256"],
        description="List of allowed signing algorithms (RS256, ES256, etc.)",
    )
    required_claims: list[str] = Field(
        default_factory=lambda: ["exp", "iat", "iss", "aud", "sub"],
        description="List of claims that must be present in the token",
    )
    jwks_cache_ttl: int = Field(
        default=3600,
        description="TTL for JWKS cache in seconds",
        ge=60,
    )
    clock_skew_seconds: float = Field(
        default=30.0,
        description="Allowed clock skew for token validation",
        ge=0.0,
    )

    _jwk_client: PyJWKClient | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _init_jwk_client(self) -> Self:
        """Initialize the JWK client after model creation."""
        jwks_url = (
            str(self.jwks_url)
            if self.jwks_url
            else f"{str(self.issuer).rstrip('/')}/.well-known/jwks.json"
        )
        self._jwk_client = PyJWKClient(jwks_url, lifespan=self.jwks_cache_ttl)
        return self

    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Authenticate using OIDC JWT validation.

        Args:
            token: The JWT to authenticate.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If authentication fails.
        """
        if self._jwk_client is None:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OIDC not initialized",
            )

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=str(self.issuer).rstrip("/"),
                leeway=self.clock_skew_seconds,
                options={
                    "require": self.required_claims,
                },
            )

            return AuthenticatedUser(
                token=token,
                scheme="oidc",
                claims=claims,
            )

        except jwt.ExpiredSignatureError:
            logger.debug(
                "OIDC authentication failed",
                extra={"reason": "token_expired", "scheme": "oidc"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            ) from None
        except jwt.InvalidAudienceError:
            logger.debug(
                "OIDC authentication failed",
                extra={"reason": "invalid_audience", "scheme": "oidc"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid token audience",
            ) from None
        except jwt.InvalidIssuerError:
            logger.debug(
                "OIDC authentication failed",
                extra={"reason": "invalid_issuer", "scheme": "oidc"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer",
            ) from None
        except jwt.MissingRequiredClaimError as e:
            logger.debug(
                "OIDC authentication failed",
                extra={"reason": "missing_claim", "claim": e.claim, "scheme": "oidc"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail=f"Missing required claim: {e.claim}",
            ) from None
        except jwt.PyJWKClientError as e:
            logger.error(
                "OIDC authentication failed",
                extra={
                    "reason": "jwks_client_error",
                    "error": str(e),
                    "scheme": "oidc",
                },
            )
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to fetch signing keys",
            ) from None
        except jwt.InvalidTokenError as e:
            logger.debug(
                "OIDC authentication failed",
                extra={"reason": "invalid_token", "error": str(e), "scheme": "oidc"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials",
            ) from None


class OAuth2ServerAuth(ServerAuthScheme):
    """OAuth2 authentication for A2A server.

    Declares OAuth2 security scheme in AgentCard and validates tokens using
    either JWKS for JWT tokens or token introspection for opaque tokens.

    This is distinct from OIDCAuth in that it declares an explicit OAuth2SecurityScheme
    with flows, rather than an OpenIdConnectSecurityScheme with discovery URL.

    Attributes:
        token_url: OAuth2 token endpoint URL for client_credentials flow.
        authorization_url: OAuth2 authorization endpoint for authorization_code flow.
        refresh_url: Optional refresh token endpoint URL.
        scopes: Available OAuth2 scopes with descriptions.
        jwks_url: JWKS URL for JWT validation. Required if not using introspection.
        introspection_url: Token introspection endpoint (RFC 7662). Alternative to JWKS.
        introspection_client_id: Client ID for introspection endpoint authentication.
        introspection_client_secret: Client secret for introspection endpoint.
        audience: Expected audience claim for JWT validation.
        issuer: Expected issuer claim for JWT validation.
        algorithms: Allowed JWT signing algorithms.
        required_claims: Claims that must be present in the token.
        jwks_cache_ttl: TTL for JWKS cache in seconds.
        clock_skew_seconds: Allowed clock skew for token validation.
    """

    token_url: HttpUrl = Field(
        description="OAuth2 token endpoint URL",
    )
    authorization_url: HttpUrl | None = Field(
        default=None,
        description="OAuth2 authorization endpoint URL for authorization_code flow",
    )
    refresh_url: HttpUrl | None = Field(
        default=None,
        description="OAuth2 refresh token endpoint URL",
    )
    scopes: dict[str, str] = Field(
        default_factory=dict,
        description="Available OAuth2 scopes with descriptions",
    )
    jwks_url: HttpUrl | None = Field(
        default=None,
        description="JWKS URL for JWT validation. Required if not using introspection.",
    )
    introspection_url: HttpUrl | None = Field(
        default=None,
        description="Token introspection endpoint (RFC 7662). Alternative to JWKS.",
    )
    introspection_client_id: str | None = Field(
        default=None,
        description="Client ID for introspection endpoint authentication",
    )
    introspection_client_secret: CoercedSecretStr | None = Field(
        default=None,
        description="Client secret for introspection endpoint authentication",
    )
    audience: str | None = Field(
        default=None,
        description="Expected audience claim for JWT validation",
    )
    issuer: str | None = Field(
        default=None,
        description="Expected issuer claim for JWT validation",
    )
    algorithms: list[str] = Field(
        default_factory=lambda: ["RS256"],
        description="Allowed JWT signing algorithms",
    )
    required_claims: list[str] = Field(
        default_factory=lambda: ["exp", "iat"],
        description="Claims that must be present in the token",
    )
    jwks_cache_ttl: int = Field(
        default=3600,
        description="TTL for JWKS cache in seconds",
        ge=60,
    )
    clock_skew_seconds: float = Field(
        default=30.0,
        description="Allowed clock skew for token validation",
        ge=0.0,
    )

    _jwk_client: PyJWKClient | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_and_init(self) -> Self:
        """Validate configuration and initialize JWKS client if needed."""
        if not self.jwks_url and not self.introspection_url:
            raise ValueError(
                "Either jwks_url or introspection_url must be provided for token validation"
            )

        if self.introspection_url:
            if not self.introspection_client_id or not self.introspection_client_secret:
                raise ValueError(
                    "introspection_client_id and introspection_client_secret are required "
                    "when using token introspection"
                )

        if self.jwks_url:
            self._jwk_client = PyJWKClient(
                str(self.jwks_url), lifespan=self.jwks_cache_ttl
            )

        return self

    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Authenticate using OAuth2 token validation.

        Uses JWKS validation if jwks_url is configured, otherwise falls back
        to token introspection.

        Args:
            token: The OAuth2 access token to authenticate.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If authentication fails.
        """
        if self._jwk_client:
            return await self._authenticate_jwt(token)
        return await self._authenticate_introspection(token)

    async def _authenticate_jwt(self, token: str) -> AuthenticatedUser:
        """Authenticate using JWKS JWT validation."""
        if self._jwk_client is None:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth2 JWKS not initialized",
            )

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)

            decode_options: dict[str, Any] = {
                "require": self.required_claims,
            }

            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.clock_skew_seconds,
                options=decode_options,
            )

            return AuthenticatedUser(
                token=token,
                scheme="oauth2",
                claims=claims,
            )

        except jwt.ExpiredSignatureError:
            logger.debug(
                "OAuth2 authentication failed",
                extra={"reason": "token_expired", "scheme": "oauth2"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
            ) from None
        except jwt.InvalidAudienceError:
            logger.debug(
                "OAuth2 authentication failed",
                extra={"reason": "invalid_audience", "scheme": "oauth2"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid token audience",
            ) from None
        except jwt.InvalidIssuerError:
            logger.debug(
                "OAuth2 authentication failed",
                extra={"reason": "invalid_issuer", "scheme": "oauth2"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer",
            ) from None
        except jwt.MissingRequiredClaimError as e:
            logger.debug(
                "OAuth2 authentication failed",
                extra={"reason": "missing_claim", "claim": e.claim, "scheme": "oauth2"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail=f"Missing required claim: {e.claim}",
            ) from None
        except jwt.PyJWKClientError as e:
            logger.error(
                "OAuth2 authentication failed",
                extra={
                    "reason": "jwks_client_error",
                    "error": str(e),
                    "scheme": "oauth2",
                },
            )
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to fetch signing keys",
            ) from None
        except jwt.InvalidTokenError as e:
            logger.debug(
                "OAuth2 authentication failed",
                extra={"reason": "invalid_token", "error": str(e), "scheme": "oauth2"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication credentials",
            ) from None

    async def _authenticate_introspection(self, token: str) -> AuthenticatedUser:
        """Authenticate using OAuth2 token introspection (RFC 7662)."""
        import httpx

        if not self.introspection_url:
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OAuth2 introspection not configured",
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    str(self.introspection_url),
                    data={"token": token},
                    auth=(
                        self.introspection_client_id or "",
                        self.introspection_client_secret.get_secret_value()
                        if self.introspection_client_secret
                        else "",
                    ),
                )
                response.raise_for_status()
                introspection_result = response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "OAuth2 introspection failed",
                extra={"reason": "http_error", "status_code": e.response.status_code},
            )
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token introspection service unavailable",
            ) from None
        except Exception as e:
            logger.error(
                "OAuth2 introspection failed",
                extra={"reason": "unexpected_error", "error": str(e)},
            )
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="Token introspection failed",
            ) from None

        if not introspection_result.get("active", False):
            logger.debug(
                "OAuth2 authentication failed",
                extra={"reason": "token_not_active", "scheme": "oauth2"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Token is not active",
            )

        return AuthenticatedUser(
            token=token,
            scheme="oauth2",
            claims=introspection_result,
        )

    def to_security_scheme(self) -> OAuth2SecurityScheme:
        """Generate OAuth2SecurityScheme for AgentCard declaration.

        Creates an OAuth2SecurityScheme with appropriate flows based on
        the configured URLs. Includes client_credentials flow if token_url
        is set, and authorization_code flow if authorization_url is set.

        Returns:
            OAuth2SecurityScheme suitable for use in AgentCard security_schemes.
        """
        from a2a.types import (
            AuthorizationCodeOAuthFlow,
            ClientCredentialsOAuthFlow,
            OAuth2SecurityScheme,
            OAuthFlows,
        )

        client_credentials = None
        authorization_code = None

        if self.token_url:
            client_credentials = ClientCredentialsOAuthFlow(
                token_url=str(self.token_url),
                refresh_url=str(self.refresh_url) if self.refresh_url else None,
                scopes=self.scopes,
            )

        if self.authorization_url:
            authorization_code = AuthorizationCodeOAuthFlow(
                authorization_url=str(self.authorization_url),
                token_url=str(self.token_url),
                refresh_url=str(self.refresh_url) if self.refresh_url else None,
                scopes=self.scopes,
            )

        return OAuth2SecurityScheme(
            flows=OAuthFlows(
                client_credentials=client_credentials,
                authorization_code=authorization_code,
            ),
            description="OAuth2 authentication",
        )


class APIKeyServerAuth(ServerAuthScheme):
    """API Key authentication for A2A server.

    Validates requests using an API key in a header, query parameter, or cookie.

    Attributes:
        name: The name of the API key parameter (default: X-API-Key).
        location: Where to look for the API key (header, query, or cookie).
        api_key: The expected API key value.
    """

    name: str = Field(
        default="X-API-Key",
        description="Name of the API key parameter",
    )
    location: Literal["header", "query", "cookie"] = Field(
        default="header",
        description="Where to look for the API key",
    )
    api_key: CoercedSecretStr = Field(
        description="Expected API key value",
    )

    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Authenticate using API key comparison.

        Args:
            token: The API key to authenticate.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If authentication fails.
        """
        if token != self.api_key.get_secret_value():
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        return AuthenticatedUser(
            token=token,
            scheme="api_key",
        )


class MTLSServerAuth(ServerAuthScheme):
    """Mutual TLS authentication marker for AgentCard declaration.

    This scheme is primarily for AgentCard security_schemes declaration.
    Actual mTLS verification happens at the TLS/transport layer, not
    at the application layer via token validation.

    When configured, this signals to clients that the server requires
    client certificates for authentication.
    """

    description: str = Field(
        default="Mutual TLS certificate authentication",
        description="Description for the security scheme",
    )

    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Return authenticated user for mTLS.

        mTLS verification happens at the transport layer before this is called.
        If we reach this point, the TLS handshake with client cert succeeded.

        Args:
            token: Certificate subject or identifier (from TLS layer).

        Returns:
            AuthenticatedUser indicating mTLS authentication.
        """
        return AuthenticatedUser(
            token=token or "mtls-verified",
            scheme="mtls",
        )

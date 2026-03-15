"""A2A authentication schemas."""

from crewai.a2a.auth.client_schemes import (
    APIKeyAuth,
    AuthScheme,
    BearerTokenAuth,
    ClientAuthScheme,
    HTTPBasicAuth,
    HTTPDigestAuth,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
    TLSConfig,
)
from crewai.a2a.auth.server_schemes import (
    AuthenticatedUser,
    OIDCAuth,
    ServerAuthScheme,
    SimpleTokenAuth,
)


__all__ = [
    "APIKeyAuth",
    "AuthScheme",
    "AuthenticatedUser",
    "BearerTokenAuth",
    "ClientAuthScheme",
    "HTTPBasicAuth",
    "HTTPDigestAuth",
    "OAuth2AuthorizationCode",
    "OAuth2ClientCredentials",
    "OIDCAuth",
    "ServerAuthScheme",
    "SimpleTokenAuth",
    "TLSConfig",
]

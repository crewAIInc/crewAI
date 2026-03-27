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
    EnterpriseTokenAuth,
    OIDCAuth,
    ServerAuthScheme,
    SimpleTokenAuth,
)
from crewai.a2a.auth.totp_scheme import TOTPClientAuthScheme, TOTPServerAuthScheme


__all__ = [
    "APIKeyAuth",
    "AuthScheme",
    "AuthenticatedUser",
    "BearerTokenAuth",
    "ClientAuthScheme",
    "EnterpriseTokenAuth",
    "HTTPBasicAuth",
    "HTTPDigestAuth",
    "OAuth2AuthorizationCode",
    "OAuth2ClientCredentials",
    "OIDCAuth",
    "ServerAuthScheme",
    "SimpleTokenAuth",
    "TLSConfig",
    "TOTPClientAuthScheme",
    "TOTPServerAuthScheme",
]

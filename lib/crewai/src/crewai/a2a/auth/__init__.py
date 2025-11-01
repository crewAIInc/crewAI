"""A2A authentication schemas."""

from crewai.a2a.auth.schemas import (
    APIKeyAuth,
    BearerTokenAuth,
    HTTPBasicAuth,
    HTTPDigestAuth,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
)


__all__ = [
    "APIKeyAuth",
    "BearerTokenAuth",
    "HTTPBasicAuth",
    "HTTPDigestAuth",
    "OAuth2AuthorizationCode",
    "OAuth2ClientCredentials",
]

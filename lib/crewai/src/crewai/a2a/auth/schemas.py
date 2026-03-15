"""Deprecated: Authentication schemes for A2A protocol agents.

This module is deprecated. Import from crewai.a2a.auth instead:
- crewai.a2a.auth.ClientAuthScheme (replaces AuthScheme)
- crewai.a2a.auth.BearerTokenAuth
- crewai.a2a.auth.HTTPBasicAuth
- crewai.a2a.auth.HTTPDigestAuth
- crewai.a2a.auth.APIKeyAuth
- crewai.a2a.auth.OAuth2ClientCredentials
- crewai.a2a.auth.OAuth2AuthorizationCode
"""

from __future__ import annotations

from typing_extensions import deprecated

from crewai.a2a.auth.client_schemes import (
    APIKeyAuth as _APIKeyAuth,
    BearerTokenAuth as _BearerTokenAuth,
    ClientAuthScheme as _ClientAuthScheme,
    HTTPBasicAuth as _HTTPBasicAuth,
    HTTPDigestAuth as _HTTPDigestAuth,
    OAuth2AuthorizationCode as _OAuth2AuthorizationCode,
    OAuth2ClientCredentials as _OAuth2ClientCredentials,
)


@deprecated("Use ClientAuthScheme from crewai.a2a.auth instead", category=FutureWarning)
class AuthScheme(_ClientAuthScheme):
    """Deprecated: Use ClientAuthScheme from crewai.a2a.auth instead."""


@deprecated("Import from crewai.a2a.auth instead", category=FutureWarning)
class BearerTokenAuth(_BearerTokenAuth):
    """Deprecated: Import from crewai.a2a.auth instead."""


@deprecated("Import from crewai.a2a.auth instead", category=FutureWarning)
class HTTPBasicAuth(_HTTPBasicAuth):
    """Deprecated: Import from crewai.a2a.auth instead."""


@deprecated("Import from crewai.a2a.auth instead", category=FutureWarning)
class HTTPDigestAuth(_HTTPDigestAuth):
    """Deprecated: Import from crewai.a2a.auth instead."""


@deprecated("Import from crewai.a2a.auth instead", category=FutureWarning)
class APIKeyAuth(_APIKeyAuth):
    """Deprecated: Import from crewai.a2a.auth instead."""


@deprecated("Import from crewai.a2a.auth instead", category=FutureWarning)
class OAuth2ClientCredentials(_OAuth2ClientCredentials):
    """Deprecated: Import from crewai.a2a.auth instead."""


@deprecated("Import from crewai.a2a.auth instead", category=FutureWarning)
class OAuth2AuthorizationCode(_OAuth2AuthorizationCode):
    """Deprecated: Import from crewai.a2a.auth instead."""


__all__ = [
    "APIKeyAuth",
    "AuthScheme",
    "BearerTokenAuth",
    "HTTPBasicAuth",
    "HTTPDigestAuth",
    "OAuth2AuthorizationCode",
    "OAuth2ClientCredentials",
]

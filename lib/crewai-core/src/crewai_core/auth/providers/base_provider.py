"""Base OAuth2 provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from crewai_core.auth.oauth2 import Oauth2Settings


class BaseProvider(ABC):
    """Abstract base class for OAuth2 providers."""

    def __init__(self, settings: Oauth2Settings):
        self.settings = settings

    @abstractmethod
    def get_authorize_url(self) -> str:
        """Return the authorization endpoint URL."""

    @abstractmethod
    def get_token_url(self) -> str:
        """Return the token endpoint URL."""

    @abstractmethod
    def get_jwks_url(self) -> str:
        """Return the JWKS endpoint URL."""

    @abstractmethod
    def get_issuer(self) -> str:
        """Return the OAuth issuer identifier."""

    @abstractmethod
    def get_audience(self) -> str:
        """Return the OAuth audience identifier."""

    @abstractmethod
    def get_client_id(self) -> str:
        """Return the OAuth client identifier."""

    def get_required_fields(self) -> list[str]:
        """Return provider-specific keys required inside ``Oauth2Settings.extra``."""
        return []

    def get_oauth_scopes(self) -> list[str]:
        """Return the OAuth scopes to request."""
        return ["openid", "profile", "email"]

"""Base OAuth2 provider interface."""

from abc import ABC, abstractmethod

from crewai.auth.oauth2 import Oauth2Settings


class BaseProvider(ABC):
    """Abstract base class for OAuth2 providers."""

    def __init__(self, settings: Oauth2Settings):
        self.settings = settings

    @abstractmethod
    def get_authorize_url(self) -> str: ...

    @abstractmethod
    def get_token_url(self) -> str: ...

    @abstractmethod
    def get_jwks_url(self) -> str: ...

    @abstractmethod
    def get_issuer(self) -> str: ...

    @abstractmethod
    def get_audience(self) -> str: ...

    @abstractmethod
    def get_client_id(self) -> str: ...

    def get_required_fields(self) -> list[str]:
        """Returns which provider-specific fields inside the "extra" dict will be required."""
        return []

    def get_oauth_scopes(self) -> list[str]:
        """Returns the OAuth scopes to request."""
        return ["openid", "profile", "email"]

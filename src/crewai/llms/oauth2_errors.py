"""OAuth2 error classes for CrewAI."""

from typing import Optional


class OAuth2Error(Exception):
    """Base exception class for OAuth2 operation errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class OAuth2ConfigurationError(OAuth2Error):
    """Exception raised for OAuth2 configuration errors."""
    pass


class OAuth2AuthenticationError(OAuth2Error):
    """Exception raised for OAuth2 authentication failures."""
    pass


class OAuth2ValidationError(OAuth2Error):
    """Exception raised for OAuth2 validation errors."""
    pass

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
import os
from typing import Any


_platform_integration_token: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("platform_integration_token", default=None)
)


def set_platform_integration_token(integration_token: str) -> None:
    """Set the platform integration token in the current context.

    Args:
        integration_token: The integration token to set.
    """
    _platform_integration_token.set(integration_token)


def get_platform_integration_token() -> str | None:
    """Get the platform integration token from the current context or environment.

    Returns:
        The integration token if set, otherwise None.
    """
    token = _platform_integration_token.get()
    if token is None:
        token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")
    return token


@contextmanager
def platform_context(integration_token: str) -> Generator[None, Any, None]:
    """Context manager to temporarily set the platform integration token.

    Args:
      integration_token: The integration token to set within the context.
    """
    token = _platform_integration_token.set(integration_token)
    try:
        yield
    finally:
        _platform_integration_token.reset(token)

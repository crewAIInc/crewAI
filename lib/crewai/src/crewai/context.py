import os
import contextvars
from typing import Optional
from contextlib import contextmanager

_platform_integration_token: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "platform_integration_token", default=None
)

def set_platform_integration_token(integration_token: str) -> None:
    _platform_integration_token.set(integration_token)

def get_platform_integration_token() -> Optional[str]:
    token = _platform_integration_token.get()
    if token is None:
        token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")
    return token

@contextmanager
def platform_context(integration_token: str):
    token = _platform_integration_token.set(integration_token)
    try:
        yield
    finally:
        _platform_integration_token.reset(token)

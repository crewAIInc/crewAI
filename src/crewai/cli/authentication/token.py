from .utils import TokenManager
from typing import Optional
from pathlib import Path

def get_auth_token() -> str:
    """Get the authentication token."""
    token_manager = TokenManager()
    return token_manager.get_token()

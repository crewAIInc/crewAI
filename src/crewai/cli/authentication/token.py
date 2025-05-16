from .utils import TokenManager
from typing import Optional
from pathlib import Path

def get_auth_token() -> str:
    """Get the authentication token."""
    access_token = TokenManager().get_token()
    if not access_token:
        raise Exception("No token found, make sure you are logged in")
    return access_token

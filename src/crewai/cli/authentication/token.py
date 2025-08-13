from .utils import TokenManager
from logging import getLogger

logger = getLogger(__name__)


def get_auth_token() -> str:
    """Get the authentication token."""
    access_token = TokenManager().get_token()
    if not access_token:
        logger.warning("No token found, make sure you are logged in")
        return ""
    return access_token

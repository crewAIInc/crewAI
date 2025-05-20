from .utils import TokenManager, get_auth_token_with_refresh_token


def get_auth_token() -> str:
    """Get the authentication token. Uses refresh token to fetch a new token if current one is expired."""
    access_token = TokenManager().get_token("access_token")
    refresh_token = TokenManager().get_token("refresh_token")

    # Token could be expired, so we use the refresh token to fetch a new one.
    # Skip if refresh token is not available.
    if not access_token and refresh_token:
        data = get_auth_token_with_refresh_token(refresh_token)
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")

        if access_token and refresh_token:
            TokenManager().save_access_token(access_token, data["expires_in"])
            TokenManager().save_refresh_token(refresh_token)

    if not access_token:
        raise Exception("Access token could not be obtained. Please sign in again.")
    return access_token

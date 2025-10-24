import asyncio
from collections.abc import Awaitable, Callable, MutableMapping

from a2a.types import (
    APIKeySecurityScheme,
    AgentCard,
    HTTPAuthSecurityScheme,
    OAuth2SecurityScheme,
)
from httpx import AsyncClient, Response

from crewai.a2a.auth.schemas import (
    APIKeyAuth,
    AuthScheme,
    BearerTokenAuth,
    HTTPBasicAuth,
    HTTPDigestAuth,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
)


def validate_auth_against_agent_card(
    agent_card: AgentCard, auth: AuthScheme | None
) -> None:
    """Validate that provided auth matches AgentCard security requirements.

    Args:
        agent_card: The A2A AgentCard containing security requirements.
        auth: User-provided authentication scheme (or None).

    Raises:
        A2AClientHTTPError: If auth doesn't match AgentCard requirements (status_code=401).
    """
    from a2a.client.errors import A2AClientHTTPError

    if not agent_card.security or not agent_card.security_schemes:
        return

    if not auth:
        msg = "AgentCard requires authentication but no auth scheme provided"
        raise A2AClientHTTPError(401, msg)

    first_security_req = agent_card.security[0] if agent_card.security else {}

    for scheme_name in first_security_req.keys():
        security_scheme_wrapper = agent_card.security_schemes.get(scheme_name)
        if not security_scheme_wrapper:
            continue

        scheme = security_scheme_wrapper.root

        if isinstance(scheme, OAuth2SecurityScheme):
            if not isinstance(
                auth,
                (OAuth2ClientCredentials, OAuth2AuthorizationCode, BearerTokenAuth),
            ):
                msg = f"AgentCard requires OAuth2 authentication, but {type(auth).__name__} was provided"
                raise A2AClientHTTPError(401, msg)
            return

        if isinstance(scheme, APIKeySecurityScheme):
            if not isinstance(auth, APIKeyAuth):
                msg = f"AgentCard requires API Key authentication, but {type(auth).__name__} was provided"
                raise A2AClientHTTPError(401, msg)
            return

        if isinstance(scheme, HTTPAuthSecurityScheme):
            http_scheme_lower = scheme.scheme.lower()

            if http_scheme_lower == "basic" and not isinstance(auth, HTTPBasicAuth):
                msg = f"AgentCard requires HTTP Basic authentication, but {type(auth).__name__} was provided"
                raise A2AClientHTTPError(401, msg)

            if http_scheme_lower == "digest" and not isinstance(auth, HTTPDigestAuth):
                msg = f"AgentCard requires HTTP Digest authentication, but {type(auth).__name__} was provided"
                raise A2AClientHTTPError(401, msg)

            if http_scheme_lower == "bearer" and not isinstance(auth, BearerTokenAuth):
                msg = f"AgentCard requires Bearer token authentication, but {type(auth).__name__} was provided"
                raise A2AClientHTTPError(401, msg)

            return

    msg = "Could not validate auth against AgentCard security requirements"
    raise A2AClientHTTPError(401, msg)


async def retry_on_401(
    request_func: Callable[[], Awaitable[Response]],
    auth_scheme: AuthScheme | None,
    client: AsyncClient,
    headers: MutableMapping[str, str],
    max_retries: int = 3,
) -> Response:
    """Retry a request on 401 authentication error.

    Handles 401 errors by:
    1. Parsing WWW-Authenticate header
    2. Re-acquiring credentials
    3. Retrying the request

    Args:
        request_func: Async function that makes the HTTP request.
        auth_scheme: Authentication scheme to refresh credentials with.
        client: HTTP client for making requests.
        headers: Request headers to update with new auth.
        max_retries: Maximum number of retry attempts (default: 3).

    Returns:
        HTTP response from the request.

    Raises:
        httpx.HTTPStatusError: If retries are exhausted or auth scheme is None.
    """
    last_response: Response | None = None

    for attempt in range(max_retries):
        response = await request_func()

        if response.status_code != 401:
            return response

        last_response = response

        if auth_scheme is None:
            response.raise_for_status()
            return response

        www_authenticate = response.headers.get("WWW-Authenticate", "")

        if attempt >= max_retries - 1:
            break

        backoff_time = 2**attempt
        await asyncio.sleep(backoff_time)

        await auth_scheme.apply_auth(client, headers)

    if last_response:
        last_response.raise_for_status()
        return last_response

    msg = "retry_on_401 failed without making any requests"
    raise RuntimeError(msg)


def configure_auth_client(
    auth: HTTPDigestAuth | APIKeyAuth, client: AsyncClient
) -> None:
    """Configure HTTP client with auth-specific settings.

    Only HTTPDigestAuth and APIKeyAuth need client configuration.

    Args:
        auth: Authentication scheme that requires client configuration.
        client: HTTP client to configure.
    """
    auth.configure_client(client)

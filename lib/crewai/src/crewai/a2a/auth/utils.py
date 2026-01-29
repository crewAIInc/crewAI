"""Authentication utilities for A2A protocol agent communication.

Provides validation and retry logic for various authentication schemes including
OAuth2, API keys, and HTTP authentication methods.
"""

import asyncio
from collections.abc import Awaitable, Callable, MutableMapping
import hashlib
import re
import threading
from typing import Final, Literal, cast

from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    APIKeySecurityScheme,
    AgentCard,
    HTTPAuthSecurityScheme,
    OAuth2SecurityScheme,
)
from httpx import AsyncClient, Response

from crewai.a2a.auth.client_schemes import (
    APIKeyAuth,
    BearerTokenAuth,
    ClientAuthScheme,
    HTTPBasicAuth,
    HTTPDigestAuth,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
)


class _AuthStore:
    """Store for authentication schemes with safe concurrent access."""

    def __init__(self) -> None:
        self._store: dict[str, ClientAuthScheme | None] = {}
        self._lock = threading.RLock()

    @staticmethod
    def compute_key(auth_type: str, auth_data: str) -> str:
        """Compute a collision-resistant key using SHA-256."""
        content = f"{auth_type}:{auth_data}"
        return hashlib.sha256(content.encode()).hexdigest()

    def set(self, key: str, auth: ClientAuthScheme | None) -> None:
        """Store an auth scheme."""
        with self._lock:
            self._store[key] = auth

    def get(self, key: str) -> ClientAuthScheme | None:
        """Retrieve an auth scheme by key."""
        with self._lock:
            return self._store.get(key)

    def __setitem__(self, key: str, value: ClientAuthScheme | None) -> None:
        with self._lock:
            self._store[key] = value

    def __getitem__(self, key: str) -> ClientAuthScheme | None:
        with self._lock:
            return self._store[key]


_auth_store = _AuthStore()

_SCHEME_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\w+)\s+(.+?)(?=,\s*\w+\s+|$)")
_PARAM_PATTERN: Final[re.Pattern[str]] = re.compile(r'(\w+)=(?:"([^"]*)"|([^\s,]+))')

_SCHEME_AUTH_MAPPING: Final[dict[type, tuple[type[ClientAuthScheme], ...]]] = {
    OAuth2SecurityScheme: (
        OAuth2ClientCredentials,
        OAuth2AuthorizationCode,
        BearerTokenAuth,
    ),
    APIKeySecurityScheme: (APIKeyAuth,),
}

_HTTPSchemeType = Literal["basic", "digest", "bearer"]

_HTTP_SCHEME_MAPPING: Final[dict[_HTTPSchemeType, type[ClientAuthScheme]]] = {
    "basic": HTTPBasicAuth,
    "digest": HTTPDigestAuth,
    "bearer": BearerTokenAuth,
}


def _raise_auth_mismatch(
    expected_classes: type[ClientAuthScheme] | tuple[type[ClientAuthScheme], ...],
    provided_auth: ClientAuthScheme,
) -> None:
    """Raise authentication mismatch error.

    Args:
        expected_classes: Expected authentication class or tuple of classes.
        provided_auth: Actually provided authentication instance.

    Raises:
        A2AClientHTTPError: Always raises with 401 status code.
    """
    if isinstance(expected_classes, tuple):
        if len(expected_classes) == 1:
            required = expected_classes[0].__name__
        else:
            names = [cls.__name__ for cls in expected_classes]
            required = f"one of ({', '.join(names)})"
    else:
        required = expected_classes.__name__

    msg = (
        f"AgentCard requires {required} authentication, "
        f"but {type(provided_auth).__name__} was provided"
    )
    raise A2AClientHTTPError(401, msg)


def parse_www_authenticate(header_value: str) -> dict[str, dict[str, str]]:
    """Parse WWW-Authenticate header into auth challenges.

    Args:
        header_value: The WWW-Authenticate header value.

    Returns:
        Dictionary mapping auth scheme to its parameters.
        Example: {"Bearer": {"realm": "api", "scope": "read write"}}
    """
    if not header_value:
        return {}

    challenges: dict[str, dict[str, str]] = {}

    for match in _SCHEME_PATTERN.finditer(header_value):
        scheme = match.group(1)
        params_str = match.group(2)

        params: dict[str, str] = {}

        for param_match in _PARAM_PATTERN.finditer(params_str):
            key = param_match.group(1)
            value = param_match.group(2) or param_match.group(3)
            params[key] = value

        challenges[scheme] = params

    return challenges


def validate_auth_against_agent_card(
    agent_card: AgentCard, auth: ClientAuthScheme | None
) -> None:
    """Validate that provided auth matches AgentCard security requirements.

    Args:
        agent_card: The A2A AgentCard containing security requirements.
        auth: User-provided authentication scheme (or None).

    Raises:
        A2AClientHTTPError: If auth doesn't match AgentCard requirements (status_code=401).
    """

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

        if allowed_classes := _SCHEME_AUTH_MAPPING.get(type(scheme)):
            if not isinstance(auth, allowed_classes):
                _raise_auth_mismatch(allowed_classes, auth)
            return

        if isinstance(scheme, HTTPAuthSecurityScheme):
            scheme_key = cast(_HTTPSchemeType, scheme.scheme.lower())
            if required_class := _HTTP_SCHEME_MAPPING.get(scheme_key):
                if not isinstance(auth, required_class):
                    _raise_auth_mismatch(required_class, auth)
            return

    msg = "Could not validate auth against AgentCard security requirements"
    raise A2AClientHTTPError(401, msg)


async def retry_on_401(
    request_func: Callable[[], Awaitable[Response]],
    auth_scheme: ClientAuthScheme | None,
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
    last_challenges: dict[str, dict[str, str]] = {}

    for attempt in range(max_retries):
        response = await request_func()

        if response.status_code != 401:
            return response

        last_response = response

        if auth_scheme is None:
            response.raise_for_status()
            return response

        www_authenticate = response.headers.get("WWW-Authenticate", "")
        challenges = parse_www_authenticate(www_authenticate)
        last_challenges = challenges

        if attempt >= max_retries - 1:
            break

        backoff_time = 2**attempt
        await asyncio.sleep(backoff_time)

        await auth_scheme.apply_auth(client, headers)

    if last_response:
        last_response.raise_for_status()
        return last_response

    msg = "retry_on_401 failed without making any requests"
    if last_challenges:
        challenge_info = ", ".join(
            f"{scheme} (realm={params.get('realm', 'N/A')})"
            for scheme, params in last_challenges.items()
        )
        msg = f"{msg}. Server challenges: {challenge_info}"
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

"""TOTP (Time-based One-Time Password) authentication for A2A protocol.

Provides second-factor authentication using TOTP codes alongside bearer tokens.
Supports both per-peer seeds (identified by bearer token) and shared seed mode.

Server-side: Validates X-TOTP header against configured seeds.
Client-side: Injects X-TOTP header on outgoing requests.

Requires: pyotp>=2.9.0
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import MutableMapping
import logging
import os
from typing import Annotated, ClassVar

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SecretStr

from crewai.a2a.auth.client_schemes import ClientAuthScheme
from crewai.a2a.auth.server_schemes import (
    HTTP_401_UNAUTHORIZED,
    AuthenticatedUser,
    HTTPException,
    ServerAuthScheme,
)


logger = logging.getLogger(__name__)


def _coerce_secret_str(v: str | SecretStr | None) -> SecretStr | None:
    """Coerce string to SecretStr."""
    if v is None or isinstance(v, SecretStr):
        return v
    return SecretStr(v)


CoercedSecretStr = Annotated[SecretStr, BeforeValidator(_coerce_secret_str)]

TOTP_HEADER = "X-TOTP"


class TOTPServerAuthScheme(ServerAuthScheme):
    """TOTP second-factor authentication for A2A server.

    Validates incoming requests by requiring both a valid bearer token and a
    valid TOTP code in the X-TOTP header. The bearer token is validated by a
    delegate ServerAuthScheme, then the TOTP code is checked against the seed
    associated with the authenticated peer.

    Supports two modes:
    - Per-peer seeds: Map bearer tokens to peer names, each with their own TOTP seed.
    - Shared seed: A single TOTP seed used for all peers.

    Fails closed: missing or invalid TOTP code always results in 401.

    Attributes:
        delegate: The underlying auth scheme that validates the bearer token.
        shared_seed: A single TOTP seed for all peers (simple deployment mode).
        peer_seeds: Per-peer TOTP seeds keyed by peer name.
        token_to_peer: Maps bearer tokens to peer names for per-peer seed lookup.
        valid_window: Number of time steps to allow for clock drift (default: 1 = ±30s).
    """

    delegate: ServerAuthScheme = Field(
        description="Underlying auth scheme for bearer token validation",
    )
    shared_seed: CoercedSecretStr | None = Field(
        default=None,
        description="Single TOTP seed for all peers. Mutually exclusive with peer_seeds.",
    )
    peer_seeds: dict[str, CoercedSecretStr] | None = Field(
        default=None,
        description="Per-peer TOTP seeds keyed by peer name.",
    )
    token_to_peer: dict[str, str] | None = Field(
        default=None,
        description="Maps bearer tokens to peer names for per-peer seed lookup.",
    )
    valid_window: int = Field(
        default=1,
        description="Number of time steps to allow for clock drift (1 = ±30s)",
        ge=0,
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    async def authenticate(self, token: str) -> AuthenticatedUser:
        """Authenticate bearer token via delegate, then validate TOTP.

        Note: This method validates the bearer token only. TOTP validation
        requires the X-TOTP header, which must be checked separately via
        authenticate_with_totp().

        Args:
            token: The bearer token to authenticate.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If authentication fails.
        """
        return await self.delegate.authenticate(token)

    async def authenticate_with_totp(self, token: str, totp_code: str | None) -> AuthenticatedUser:
        """Authenticate bearer token and validate TOTP code.

        Args:
            token: The bearer token to authenticate.
            totp_code: The TOTP code from the X-TOTP header.

        Returns:
            AuthenticatedUser on successful authentication.

        Raises:
            HTTPException: If bearer token or TOTP validation fails.
        """
        import pyotp

        # First validate the bearer token via delegate
        user = await self.delegate.authenticate(token)

        # TOTP code is required
        if not totp_code:
            logger.debug(
                "TOTP authentication failed",
                extra={"reason": "missing_totp_header", "scheme": "totp"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Missing X-TOTP header",
            )

        # Resolve the seed
        seed = self._resolve_seed(token)
        if seed is None:
            logger.warning(
                "TOTP authentication failed",
                extra={"reason": "no_seed_configured", "scheme": "totp"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="TOTP not configured for this peer",
            )

        # Validate the TOTP code
        totp = pyotp.TOTP(seed)
        if not totp.verify(totp_code, valid_window=self.valid_window):
            logger.debug(
                "TOTP authentication failed",
                extra={"reason": "invalid_totp_code", "scheme": "totp"},
            )
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP code",
            )

        return AuthenticatedUser(
            token=user.token,
            scheme="totp",
            claims=user.claims,
        )

    def _resolve_seed(self, token: str) -> str | None:
        """Resolve the TOTP seed for the given token.

        Checks per-peer seeds first (via token_to_peer mapping), then
        falls back to shared_seed.

        Args:
            token: The bearer token to look up.

        Returns:
            The TOTP seed string, or None if not configured.
        """
        # Try per-peer lookup
        if self.peer_seeds and self.token_to_peer:
            peer = self.token_to_peer.get(token)
            if peer and peer in self.peer_seeds:
                return self.peer_seeds[peer].get_secret_value()

        # Fall back to shared seed
        if self.shared_seed:
            return self.shared_seed.get_secret_value()

        return None


class TOTPClientAuthScheme(ClientAuthScheme):
    """TOTP client authentication scheme for A2A protocol.

    Injects an X-TOTP header with a valid TOTP code on every outgoing request.
    Designed to be used alongside a bearer token auth scheme.

    Attributes:
        delegate: The underlying client auth scheme (e.g., BearerTokenAuth).
        seed: TOTP seed for code generation. Falls back to A2A_TOTP_SEED env var.
    """

    delegate: ClientAuthScheme = Field(
        description="Underlying client auth scheme for bearer token injection",
    )
    seed: CoercedSecretStr | None = Field(
        default=None,
        description="TOTP seed for code generation. Falls back to A2A_TOTP_SEED env var.",
    )

    def _get_seed(self) -> str | None:
        """Get the TOTP seed value."""
        if self.seed:
            return self.seed.get_secret_value()
        return os.environ.get("A2A_TOTP_SEED")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply delegate auth and inject X-TOTP header.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with authentication and TOTP code applied.

        Raises:
            ValueError: If no TOTP seed is configured.
        """
        import pyotp

        # Apply delegate auth first (e.g., bearer token)
        headers = await self.delegate.apply_auth(client, headers)

        # Generate and inject TOTP code
        seed = self._get_seed()
        if seed is None:
            raise ValueError(
                "No TOTP seed configured. Set seed parameter or A2A_TOTP_SEED env var."
            )

        totp = pyotp.TOTP(seed)
        headers[TOTP_HEADER] = totp.now()

        return headers


# ---------------------------------------------------------------------------
# CallContextBuilder integration
# ---------------------------------------------------------------------------

try:
    from a2a.server.apps.jsonrpc import CallContextBuilder
    from a2a.server.context import ServerCallContext, User
    from starlette.requests import Request

    class _TOTPAuthenticatedUser(User):
        """Authenticated user representation for TOTP-validated requests."""

        def __init__(self, user: AuthenticatedUser) -> None:
            self._user = user

        @property
        def is_authenticated(self) -> bool:
            return True

        @property
        def user_name(self) -> str:
            return self._user.token or ""

    class TOTPCallContextBuilder(CallContextBuilder):
        """CallContextBuilder that validates TOTP alongside bearer token auth.

        Extracts the bearer token from the Authorization header and the TOTP
        code from the X-TOTP header, then validates both via the underlying
        TOTPServerAuthScheme.

        The ``build()`` method is synchronous (per the CallContextBuilder
        interface) but the auth scheme is async. This is bridged via
        ``asyncio.run()`` in a fresh event loop when no loop is running, or
        via a new thread when called from within an existing async context.

        Attributes:
            auth_scheme: The TOTP server auth scheme to validate against.
        """

        def __init__(self, auth_scheme: TOTPServerAuthScheme) -> None:
            self.auth_scheme = auth_scheme

        def build(self, request: Request) -> ServerCallContext:
            """Build a ServerCallContext by validating bearer + TOTP headers.

            Args:
                request: The incoming Starlette request.

            Returns:
                ServerCallContext with authenticated user on success.

            Raises:
                HTTPException: 401 on missing/invalid credentials.
            """
            # Extract bearer token
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                logger.debug(
                    "TOTP context build failed",
                    extra={"reason": "missing_authorization_header", "scheme": "totp"},
                )
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Missing or invalid Authorization header",
                )
            token = auth_header[7:]  # Strip "Bearer "

            # Extract TOTP code
            totp_code = request.headers.get(TOTP_HEADER)

            # Run async authentication in sync context
            authenticated_user = self._run_async(
                self.auth_scheme.authenticate_with_totp(token, totp_code)
            )

            return ServerCallContext(
                user=_TOTPAuthenticatedUser(authenticated_user),
            )

        @staticmethod
        def _run_async(coro):  # noqa: ANN001, ANN205
            """Run an async coroutine from a synchronous context.

            Uses ``asyncio.run()`` when no event loop is running.
            Falls back to running in a separate thread when called
            from within an existing async event loop.
            """
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — safe to use asyncio.run
                return asyncio.run(coro)

            # Already inside an async context — run in a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()

except ImportError:
    # a2a-sdk not installed — CallContextBuilder unavailable
    pass

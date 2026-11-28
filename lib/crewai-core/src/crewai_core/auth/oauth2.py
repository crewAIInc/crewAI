"""OAuth2 device-flow authentication for the CrewAI platform."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast
import webbrowser

import httpx
from pydantic import BaseModel, Field
from rich.console import Console

from crewai_core.auth.utils import validate_jwt_token
from crewai_core.plus_api import PlusAPI
from crewai_core.settings import Settings
from crewai_core.token_manager import TokenManager


console = Console()
logger = logging.getLogger(__name__)

# RFC 8628 makes the device code's `expires_in` required; this covers a
# self-hosted provider that omits it anyway.
_FALLBACK_DEVICE_CODE_LIFETIME_SECONDS = 600

# Renew a little before expiry so a request started on the old token still lands.
REFRESH_MARGIN_SECONDS = 60

# Short: a refresh runs on its caller's way to AMP, inside a crew run or a CLI command.
_REFRESH_TIMEOUT_SECONDS = 5

TOauth2Settings = TypeVar("TOauth2Settings", bound="Oauth2Settings")


# Not named for "auth" or "code": CodeQL's sensitive-data heuristic reads such a call's
# result as a password and flags the log line that prints it.
def _refresh_rejection_reason(response: httpx.Response) -> str | None:
    """Return the RFC 6749 section 5.2 `error` of a token-endpoint reply, if it is a standard one."""
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, dict) else None
    # Only the RFC 6749 section 5.2 codes come back, as these are logged: anything else in
    # `error` is server text, and a refresh token fits any "looks like a code" pattern.
    if error in (
        "invalid_request",
        "invalid_client",
        "invalid_grant",
        "unauthorized_client",
        "unsupported_grant_type",
        "invalid_scope",
    ):
        return str(error)
    return None


class Oauth2Settings(BaseModel):
    """OAuth2 provider configuration."""

    provider: str = Field(
        description="OAuth2 provider used for authentication (e.g., workos, okta, auth0)."
    )
    client_id: str = Field(
        description="OAuth2 client ID issued by the provider, used during authentication requests."
    )
    domain: str = Field(
        description="OAuth2 provider's domain (e.g., your-org.auth0.com) used for issuing tokens."
    )
    audience: str | None = Field(
        description="OAuth2 audience value, typically used to identify the target API or resource.",
        default=None,
    )
    extra: dict[str, Any] = Field(
        description="Extra configuration for the OAuth2 provider.",
        default={},
    )

    @classmethod
    def from_settings(cls: type[TOauth2Settings]) -> TOauth2Settings:
        """Build an ``Oauth2Settings`` instance from the persisted CrewAI settings."""
        settings = Settings()

        return cls(
            provider=settings.oauth2_provider,
            domain=settings.oauth2_domain,
            client_id=settings.oauth2_client_id,
            audience=settings.oauth2_audience,
            extra=settings.oauth2_extra,
        )


if TYPE_CHECKING:
    from crewai_core.auth.providers.base_provider import BaseProvider


class ProviderFactory:
    """Factory for resolving the configured OAuth2 provider."""

    @classmethod
    def from_settings(
        cls: type["ProviderFactory"],  # noqa: UP037
        settings: Oauth2Settings | None = None,
    ) -> "BaseProvider":  # noqa: UP037
        """Create a provider instance from settings, importing the module dynamically."""
        settings = settings or Oauth2Settings.from_settings()

        import importlib

        module = importlib.import_module(
            f"crewai_core.auth.providers.{settings.provider.lower()}"
        )
        provider = getattr(
            module,
            f"{''.join(word.capitalize() for word in settings.provider.split('_'))}Provider",
        )

        return cast("BaseProvider", provider(settings))


class AuthenticationCommand:
    """Drives the OAuth2 device-flow login against the configured provider."""

    def __init__(self) -> None:
        self.token_manager = TokenManager()
        self.oauth2_provider = ProviderFactory.from_settings()

    def login(self) -> None:
        """Sign in to the CrewAI platform via the OAuth2 device flow.

        Raises:
            SystemExit: With code 1 when the device code expires before the
                user finishes signing in.
        """
        console.print("Signing in to CrewAI AMP...\n", style="bold blue")

        device_code_data = self._get_device_code()
        self._display_auth_instructions(device_code_data)

        return self._poll_for_token(device_code_data)

    def _get_device_code(self) -> dict[str, Any]:
        """Request a device code from the provider."""
        device_code_payload = {
            "client_id": self.oauth2_provider.get_client_id(),
            "scope": " ".join(self.oauth2_provider.get_oauth_scopes()),
            "audience": self.oauth2_provider.get_audience(),
        }
        response = httpx.post(
            url=self.oauth2_provider.get_authorize_url(),
            data=device_code_payload,
            timeout=20,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    def _display_auth_instructions(self, device_code_data: dict[str, str]) -> None:
        """Print and open the verification URL the user must visit."""
        verification_uri = device_code_data.get(
            "verification_uri_complete", device_code_data.get("verification_uri", "")
        )

        console.print("1. Navigate to: ", verification_uri)
        console.print("2. Enter the following code: ", device_code_data["user_code"])
        webbrowser.open(verification_uri)

    def _poll_for_token(self, device_code_data: dict[str, Any]) -> None:
        """Poll the token endpoint until authentication completes or the code expires."""
        token_payload = {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "device_code": device_code_data["device_code"],
            "client_id": self.oauth2_provider.get_client_id(),
        }

        console.print("\nWaiting for authentication... ", style="bold blue", end="")

        deadline = time.monotonic() + device_code_data.get(
            "expires_in", _FALLBACK_DEVICE_CODE_LIFETIME_SECONDS
        )
        while time.monotonic() < deadline:
            response = httpx.post(
                self.oauth2_provider.get_token_url(), data=token_payload, timeout=30
            )
            token_data = response.json()

            if response.status_code == 200:
                self._validate_and_save_token(token_data)

                console.print(
                    "Success!",
                    style="bold green",
                )

                self._post_login()

                console.print("\n[bold green]Welcome to CrewAI AMP![/bold green]\n")
                return

            # The provider's clock for the code started before ours did, so its
            # `expired_token` can arrive before our deadline; it is the same timeout.
            if token_data["error"] == "expired_token":
                break
            if token_data["error"] not in ("authorization_pending", "slow_down"):
                raise httpx.HTTPError(
                    token_data.get("error_description") or token_data.get("error")
                )

            time.sleep(device_code_data["interval"])

        console.print(
            "Timeout: the login code expired before sign-in finished. "
            "Please try again.",
            style="bold red",
        )
        raise SystemExit(1)

    def refresh_access_token(self) -> str | None:
        """Exchange the saved refresh token for a new access token and save it.

        Returns:
            The new access token, or None when there is nothing to refresh or the
            refresh failed. A refresh token the provider rejects as
            ``invalid_grant`` is discarded with the rest of the login, so the user
            is logged out. Any other rejection (such as ``invalid_client``), a
            network error, a server error or an unreadable reply keeps it for the
            next attempt, and a rotated refresh token is kept even when its access
            token fails validation. A reply lost after the provider rotated the token cannot
            be recovered: the next attempt is rejected and the user logs in again.
            A login idle for over 72 hours, or started over 30 days ago, is not
            refreshed: the token manager no longer returns its refresh token.
        """
        with self.token_manager.lock():
            # Another process may have refreshed while this one waited for the lock.
            if access_token := self.token_manager.get_token(REFRESH_MARGIN_SECONDS):
                return access_token

            saved = self.token_manager.get_refresh_token()
            if saved is None:
                return None
            refresh_token, token_url = saved
            # The refresh token only goes back to the endpoint that issued it, never
            # to one a changed or project-local settings file now names.
            if token_url != self.oauth2_provider.get_token_url():
                return None

            try:
                # Through the proxy and CA the machine had at startup: httpx.post would
                # read a project's .env (HTTPS_PROXY, SSL_CERT_FILE) loaded since.
                with PlusAPI._http_client() as client:
                    response = client.post(
                        token_url,
                        data={
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": self.oauth2_provider.get_client_id(),
                        },
                        timeout=_REFRESH_TIMEOUT_SECONDS,
                    )
            except httpx.HTTPError as e:
                logger.debug(
                    "Could not renew the saved CrewAI login: %s", type(e).__name__
                )
                return None

            # RFC 6749 section 5.2: only invalid_grant says the refresh token itself is
            # dead. invalid_client, invalid_request and the like point at settings or
            # the provider, and a fix there makes the saved login usable again.
            if response.status_code in (400, 401):
                error = _refresh_rejection_reason(response)
                if error == "invalid_grant":
                    self.token_manager.clear_tokens()
                    logger.warning(
                        "The saved CrewAI login was rejected (invalid_grant). "
                        "Run `crewai login` to sign in again."
                    )
                else:
                    logger.warning(
                        "Could not renew the saved CrewAI login: the login provider "
                        "answered HTTP %s (%s). The login is kept.",
                        response.status_code,
                        error or "no recognized error code",
                    )
                return None
            if response.status_code != 200:
                logger.debug(
                    "Could not renew the saved CrewAI login: HTTP %s",
                    response.status_code,
                )
                return None

            try:
                token_data = response.json()
            # A captive portal or proxy page can answer 200 with HTML.
            except ValueError:
                return None
            if not isinstance(token_data, dict):
                return None
            # Providers that do not rotate refresh tokens omit it from the response.
            rotated = token_data.get("refresh_token")
            if not isinstance(rotated, str):
                rotated = refresh_token
            try:
                self._validate_and_save_token(
                    {**token_data, "refresh_token": rotated},
                    jwks_timeout=_REFRESH_TIMEOUT_SECONDS,
                    keep_login_time=True,
                )
            # validate_jwt_token raises a bare Exception for every invalid token.
            except Exception:
                if rotated != refresh_token:
                    # WorkOS rotates a public client's refresh token on every
                    # exchange, so the one just sent is spent. Keep the new one,
                    # with no usable access token beside it.
                    self.token_manager.save_tokens(
                        "",
                        0,
                        refresh_token=rotated,
                        token_url=token_url,
                        keep_login_time=True,
                    )
                return None
            return cast(str, token_data["access_token"])

    def _validate_and_save_token(
        self,
        token_data: dict[str, Any],
        jwks_timeout: float = 30,
        keep_login_time: bool = False,
    ) -> None:
        """Validate the JWT and persist it, with any refresh token, via the token manager."""
        jwt_token = token_data["access_token"]
        issuer = self.oauth2_provider.get_issuer()
        jwt_token_data = {
            "jwt_token": jwt_token,
            "jwks_url": self.oauth2_provider.get_jwks_url(),
            "issuer": issuer,
            "audience": self.oauth2_provider.get_audience(),
        }

        decoded_token = validate_jwt_token(**jwt_token_data, timeout=jwks_timeout)

        expires_at = decoded_token.get("exp", 0)
        self.token_manager.save_tokens(
            jwt_token,
            expires_at,
            refresh_token=token_data.get("refresh_token"),
            token_url=self.oauth2_provider.get_token_url(),
            keep_login_time=keep_login_time,
        )

    def _post_login(self) -> None:
        """Hook called after a successful login. Override to extend behavior."""

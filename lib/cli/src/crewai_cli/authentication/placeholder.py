"""Temporary CrewAI AMP accounts for commands that need a login and find none.

A coding agent, or a person at a terminal, running `crewai deploy` or
`crewai eval --models` without `crewai login` gets an account AMP creates on
the spot. It is deleted 7 days after creation unless claimed at its claim link;
`crewai login` replaces it. CI (no coding agent, no terminal) gets none.

Kept apart from the login, in `placeholder.enc` rather than `tokens.enc`, and
sent only to the AMP that issued it. A real login always wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sys
from typing import Literal
from urllib.parse import urlparse

from crewai_core.runtime_env import detect_coding_agent
from crewai_core.token_manager import TokenManager
from cryptography.fernet import InvalidToken
import httpx
from rich.console import Console
from rich.text import Text

from crewai_cli.plus_api import PlusAPI
from crewai_cli.utils import is_dmn_mode_enabled


PLACEHOLDER_FILE = "placeholder.enc"
PlaceholderSource = Literal["deploy", "eval"]

CREATED = "Created a temporary CrewAI account for this project; it expires on {date}."
KEEP_IT = (
    "Keep it by signing up at {claim_url}, or run `crewai login` to use your "
    "own account instead."
)
USED = (
    "This used a temporary CrewAI account that expires on {date}. "
    "Keep it by signing up at {claim_url}."
)
NOT_CREATED = "A temporary CrewAI account could not be created: {reason}."
REJECTED = (
    "CrewAI AMP no longer accepts the temporary account (it expired or was "
    "removed), so it was discarded."
)
_REFUSALS = {
    429: "the limit for new temporary accounts was reached, try again later",
    503: "temporary accounts are not available right now",
    403: "temporary accounts are not available from this network",
}

console = Console()


@dataclass(frozen=True)
class Placeholder:
    """A temporary account: its AMP token and where it may be sent."""

    token: str
    organization_uuid: str
    expires_at: datetime
    claim_url: str
    origin: str


def _store() -> TokenManager:
    return TokenManager(file_path=PLACEHOLDER_FILE)


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}".lower()


def saved_placeholder(base_url: str) -> Placeholder | None:
    """The saved temporary account, when BASE_URL's AMP issued it and it has not expired."""
    try:
        # TokenManager drops it once `expires_at` has passed.
        stored = _store().get_token()
    except InvalidToken:
        # Saved under a `secret.key` that has since changed: unreadable, so as
        # good as absent. The next one created overwrites it.
        return None
    if stored is None:
        return None
    fields = json.loads(stored)
    if fields["origin"] != _origin(base_url):
        return None
    return Placeholder(
        token=fields["token"],
        organization_uuid=fields["organization_uuid"],
        expires_at=datetime.fromisoformat(fields["expires_at"]),
        claim_url=fields["claim_url"],
        origin=fields["origin"],
    )


def forget_placeholder() -> None:
    """Drop the saved temporary account (no-op when there is none)."""
    _store().clear_tokens()


def discard_rejected_placeholder() -> None:
    """AMP answered 401 to the temporary account: say so and drop it."""
    forget_placeholder()
    console.print(Text(REJECTED), style="yellow")


def may_create_placeholder() -> bool:
    """A coding agent, or a person at a terminal. Never CI: nobody there would
    see the claim link, and the account would be deleted unclaimed."""
    return detect_coding_agent() != "unknown" or (
        not is_dmn_mode_enabled() and sys.stdin.isatty()
    )


def placeholder_for(base_url: str, *, source: PlaceholderSource) -> Placeholder | None:
    """The temporary account to use with BASE_URL's AMP: the saved one, else a
    new one when this caller may have one. None means "log in first"."""
    saved = saved_placeholder(base_url)
    if saved is not None or not may_create_placeholder():
        return saved
    return create_placeholder(base_url, source=source)


def create_placeholder(
    base_url: str, *, source: PlaceholderSource
) -> Placeholder | None:
    """Ask BASE_URL's AMP for a temporary account, save it and say so; None,
    with one line saying why, when AMP does not create one."""
    client = PlusAPI(base_url=base_url)
    # No credential goes with the request, so no organization either.
    client.headers.pop("X-Crewai-Organization-Id", None)
    try:
        response = client.create_placeholder_account(
            source=source, coding_agent=detect_coding_agent()
        )
    except httpx.HTTPError as error:
        _say_not_created(f"could not reach CrewAI AMP ({type(error).__name__})")
        return None

    placeholder = _created(response, _origin(base_url))
    if placeholder is None:
        _say_not_created(
            _REFUSALS.get(response.status_code)
            or (
                "CrewAI AMP's answer was not a usable account"
                if response.status_code == 201
                else f"CrewAI AMP answered {response.status_code}"
            )
        )
        return None

    _store().save_tokens(
        json.dumps(
            {
                "token": placeholder.token,
                "organization_uuid": placeholder.organization_uuid,
                "expires_at": placeholder.expires_at.isoformat(),
                "claim_url": placeholder.claim_url,
                "origin": placeholder.origin,
            }
        ),
        int(placeholder.expires_at.timestamp()),
    )
    date = _date(placeholder)
    console.print(Text(CREATED.format(date=date)), style="green")
    console.print(Text(KEEP_IT.format(claim_url=placeholder.claim_url)))
    return placeholder


def placeholder_client(placeholder: Placeholder, base_url: str) -> PlusAPI:
    """A client for BASE_URL that carries the temporary account."""
    if _origin(base_url) != placeholder.origin:
        raise ValueError("The temporary account was issued by another CrewAI AMP")
    return PlusAPI(
        api_key=placeholder.token,
        base_url=base_url,
        organization_id=placeholder.organization_uuid,
    )


def remind_placeholder(placeholder: Placeholder) -> None:
    """After a deploy or evaluation that used it: when it expires and how to keep it."""
    console.print(
        Text(USED.format(date=_date(placeholder), claim_url=placeholder.claim_url)),
        style="yellow",
    )


def _say_not_created(reason: str) -> None:
    console.print(Text(NOT_CREATED.format(reason=reason)), style="yellow")


def _date(placeholder: Placeholder) -> str:
    return placeholder.expires_at.astimezone(timezone.utc).date().isoformat()


def _created(response: httpx.Response, origin: str) -> Placeholder | None:
    """AMP's 201 as a temporary account, or None for any other answer."""
    if response.status_code != 201:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    token = body.get("token")
    organization_uuid = body.get("organization_uuid")
    claim_url = body.get("claim_url")
    expires_at = _utc(body.get("expires_at"))
    if not (
        isinstance(token, str)
        and token
        and isinstance(organization_uuid, str)
        and organization_uuid
        and isinstance(claim_url, str)
        and _web_address(claim_url)
        and expires_at is not None
        and expires_at > datetime.now(timezone.utc)
    ):
        return None
    return Placeholder(
        token=token,
        organization_uuid=organization_uuid,
        expires_at=expires_at,
        claim_url=claim_url,
        origin=origin,
    )


def _utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        # Python 3.10's fromisoformat does not read a trailing "Z".
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _web_address(url: str) -> bool:
    """Printed for a person to open: an http(s) address with no spaces or
    control characters to restyle the terminal."""
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"https", "http"} and bool(parsed.netloc)

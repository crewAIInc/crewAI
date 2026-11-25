"""What the device-flow login tells the person before the browser opens."""

from __future__ import annotations

from typing import Any

from crewai_core.auth import oauth2
from crewai_core.auth.oauth2 import AuthenticationCommand
import pytest


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []
    monkeypatch.setattr(oauth2.webbrowser, "open", urls.append)
    return urls


def _instructions(capsys: pytest.CaptureFixture[str], data: dict[str, Any]) -> str:
    # The instructions read no settings or provider: skip __init__'s, which do.
    AuthenticationCommand._display_auth_instructions(
        AuthenticationCommand.__new__(AuthenticationCommand), data
    )
    return " ".join(capsys.readouterr().out.split())


def test_a_prefilled_link_says_the_code_is_already_filled_in(capsys, opened) -> None:
    said = _instructions(
        capsys,
        {
            "verification_uri": "https://login.example/activate",
            "verification_uri_complete": "https://login.example/activate?code=ABCD-EFGH",
            "user_code": "ABCD-EFGH",
            "expires_in": 900,
        },
    )

    assert (
        "1. Open this page (your browser should open it for you): "
        "https://login.example/activate?code=ABCD-EFGH" in said
    )
    assert "2. Check that it shows this code, already filled in: ABCD-EFGH" in said
    assert "Enter this code" not in said
    assert "The code expires in 15 minutes. Press Ctrl-C to cancel." in said
    assert opened == ["https://login.example/activate?code=ABCD-EFGH"]


def test_a_bare_link_asks_for_the_code_to_be_entered(capsys, opened) -> None:
    said = _instructions(
        capsys,
        {
            "verification_uri": "https://login.example/activate",
            "user_code": "ABCD-EFGH",
            "expires_in": 300,
        },
    )

    assert "https://login.example/activate" in said
    assert "2. Enter this code on that page: ABCD-EFGH" in said
    assert "already filled in" not in said
    assert "The code expires in 5 minutes." in said
    assert opened == ["https://login.example/activate"]


@pytest.mark.parametrize(
    ("data", "expiry"),
    [
        ({}, "The code expires in 10 minutes."),
        ({"expires_in": 60}, "The code expires in 1 minute."),
        ({"expires_in": 20}, "The code expires in 1 minute."),
        ({"expires_in": 1799}, "The code expires in 30 minutes."),
    ],
)
def test_the_time_left_is_said_in_minutes(capsys, opened, data, expiry) -> None:
    """Without `expires_in`, the same ten minutes the poll waits for."""
    said = _instructions(
        capsys,
        {
            "verification_uri": "https://login.example/activate",
            "user_code": "C",
            **data,
        },
    )

    assert f"{expiry} Press Ctrl-C to cancel." in said

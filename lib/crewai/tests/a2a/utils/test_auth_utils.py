"""Tests for WWW-Authenticate header parsing in the A2A auth utilities."""

from __future__ import annotations

import pytest

from crewai.a2a.auth.utils import parse_www_authenticate


def test_single_challenge_with_quoted_and_token_params() -> None:
    header = 'Bearer realm="api", scope="read write", error=invalid_token'

    assert parse_www_authenticate(header) == {
        "Bearer": {
            "realm": "api",
            "scope": "read write",
            "error": "invalid_token",
        }
    }


def test_multiple_challenges() -> None:
    header = 'Basic realm="x", Bearer realm="y", scope="a"'

    assert parse_www_authenticate(header) == {
        "Basic": {"realm": "x"},
        "Bearer": {"realm": "y", "scope": "a"},
    }


def test_comma_inside_quoted_value_stays_in_the_value() -> None:
    header = (
        'Bearer realm="api", error="invalid_token", '
        'error_description="The access token, which expired, is invalid"'
    )

    assert parse_www_authenticate(header) == {
        "Bearer": {
            "realm": "api",
            "error": "invalid_token",
            "error_description": "The access token, which expired, is invalid",
        }
    }


def test_escaped_quote_inside_quoted_value() -> None:
    header = r'Bearer realm="say \"hi\""'

    assert parse_www_authenticate(header) == {"Bearer": {"realm": 'say "hi"'}}


def test_empty_quoted_value_is_an_empty_string() -> None:
    assert parse_www_authenticate('Bearer realm="api", scope=""') == {
        "Bearer": {"realm": "api", "scope": ""}
    }


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer", {"Bearer": {}}),
        ('Bearer, Basic realm="x"', {"Bearer": {}, "Basic": {"realm": "x"}}),
        ('Negotiate, Bearer realm="x"', {"Negotiate": {}, "Bearer": {"realm": "x"}}),
    ],
)
def test_challenge_without_params_is_kept(
    header: str, expected: dict[str, dict[str, str]]
) -> None:
    assert parse_www_authenticate(header) == expected


@pytest.mark.parametrize("header", ["", "   "])
def test_empty_header(header: str) -> None:
    assert parse_www_authenticate(header) == {}

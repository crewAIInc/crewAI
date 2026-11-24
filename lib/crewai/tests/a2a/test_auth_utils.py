"""Tests for validating client auth against an AgentCard's security requirements."""

from __future__ import annotations

from typing import Any

import pytest
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    SecurityScheme,
)

from crewai.a2a.auth.client_schemes import APIKeyAuth, BearerTokenAuth
from crewai.a2a.auth.utils import validate_auth_against_agent_card


def _card(
    security: list[dict[str, list[str]]],
    schemes: dict[str, SecurityScheme],
) -> AgentCard:
    return AgentCard(
        name="Test Agent",
        description="Test agent",
        url="http://localhost:9999",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
        security=security,
        security_schemes=schemes,
    )


def _api_key_scheme() -> SecurityScheme:
    return SecurityScheme(
        root=APIKeySecurityScheme(in_="header", name="X-API-Key")
    )


def _bearer_scheme() -> SecurityScheme:
    return SecurityScheme(root=HTTPAuthSecurityScheme(scheme="bearer"))


def _schemes() -> dict[str, SecurityScheme]:
    return {
        "api_key_scheme": _api_key_scheme(),
        "bearer_scheme": _bearer_scheme(),
    }


def _api_key_auth() -> Any:
    return APIKeyAuth(api_key="secret")


def test_single_requirement_accepts_matching_auth() -> None:
    card = _card([{"api_key_scheme": []}], _schemes())

    validate_auth_against_agent_card(card, _api_key_auth())


def test_all_schemes_in_one_requirement_are_required() -> None:
    """A requirement with several schemes is an AND, not an OR."""
    card = _card([{"api_key_scheme": [], "bearer_scheme": []}], _schemes())

    with pytest.raises(A2AClientHTTPError, match="BearerTokenAuth"):
        validate_auth_against_agent_card(card, _api_key_auth())


def test_later_security_alternative_is_accepted() -> None:
    """Any satisfied alternative in the security list is enough."""
    card = _card([{"bearer_scheme": []}, {"api_key_scheme": []}], _schemes())

    validate_auth_against_agent_card(card, _api_key_auth())


def test_bearer_auth_matches_http_bearer_scheme() -> None:
    card = _card([{"bearer_scheme": []}], _schemes())

    validate_auth_against_agent_card(card, BearerTokenAuth(token="token"))


def test_missing_auth_is_rejected_when_required() -> None:
    card = _card([{"api_key_scheme": []}], _schemes())

    with pytest.raises(A2AClientHTTPError, match="no auth scheme"):
        validate_auth_against_agent_card(card, None)


def test_empty_requirement_means_auth_is_optional() -> None:
    card = _card([{}], _schemes())

    validate_auth_against_agent_card(card, None)

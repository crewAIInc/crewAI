"""Behavioral coverage for A2A content type negotiation."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from crewai.a2a.utils.content_type import (
    ContentTypeNegotiationError,
    negotiate_content_types,
)
import pytest


@pytest.mark.parametrize("skill_specific", [False, True])
@pytest.mark.parametrize(
    "client_modes,server_modes,expected",
    [
        (["image/*"], ["image/png", "image/jpeg"], ["image/png", "image/jpeg"]),
        (
            ["*/*"],
            ["text/plain", "application/json", "image/png"],
            ["text/plain", "application/json", "image/png"],
        ),
        (
            ["image/jpeg", "image/*"],
            ["image/png", "image/jpeg", "image/png", "text/plain"],
            ["image/jpeg", "image/png"],
        ),
        (
            ["*/*"],
            ["image/png", "image/*", "image/jpeg"],
            ["image/png", "image/jpeg"],
        ),
        (
            ["image/*"],
            ["image/png", "*/*", "image/jpeg"],
            ["image/png", "image/jpeg"],
        ),
        (
            ["image/png", "image/*"],
            ["image/png", "*/*", "image/jpeg"],
            ["image/png", "image/jpeg"],
        ),
        (["image/png"], ["image/*", "image/png"], ["image/png"]),
        (["image/*"], ["image/*", "image/png"], ["image/*"]),
        (["image/*"], ["text/plain"], []),
        ([], ["image/png"], []),
    ],
)
def test_negotiated_modes_preserve_compatible_formats(
    client_modes: list[str],
    server_modes: list[str],
    expected: list[str],
    skill_specific: bool,
) -> None:
    card = AgentCard(
        name="Media agent",
        description="Accepts and produces multiple media formats",
        url="https://example.com/a2a",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["application/pdf"] if skill_specific else server_modes,
        default_output_modes=["application/pdf"] if skill_specific else server_modes,
        skills=[
            AgentSkill(
                id="media",
                name="Media conversion",
                description="Convert media",
                tags=["media"],
                input_modes=server_modes,
                output_modes=server_modes,
            )
        ]
        if skill_specific
        else [],
    )
    original_client_modes = client_modes.copy()
    original_card = card.model_dump()

    result = negotiate_content_types(
        card,
        client_input_modes=client_modes,
        client_output_modes=client_modes,
        skill_name="media" if skill_specific else None,
        emit_event=False,
    )

    assert result.input_modes == expected
    assert result.output_modes == expected
    assert client_modes == original_client_modes
    assert card.model_dump() == original_card


def test_strict_negotiation_rejects_incompatible_output() -> None:
    card = AgentCard(
        name="Text agent",
        description="Accepts and produces text",
        url="https://example.com/a2a",
        version="1.0.0",
        capabilities=AgentCapabilities(),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[],
    )

    with pytest.raises(ContentTypeNegotiationError) as error:
        negotiate_content_types(
            card,
            client_input_modes=["text/plain"],
            client_output_modes=["image/*"],
            emit_event=False,
            strict=True,
        )

    assert error.value.direction == "output"

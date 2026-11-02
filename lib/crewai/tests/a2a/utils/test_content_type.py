"""Tests for A2A content type validation."""

from __future__ import annotations

from a2a.types import FilePart, FileWithBytes, Part, TextPart

from crewai.a2a.utils.content_type import (
    validate_content_type,
    validate_message_parts,
)


class TestValidateContentType:
    def test_exact_match_is_allowed(self) -> None:
        assert validate_content_type("text/plain", ["text/plain"])

    def test_mismatch_is_rejected(self) -> None:
        assert not validate_content_type("image/png", ["text/plain"])

    def test_wildcard_mode_matches_concrete_type(self) -> None:
        assert validate_content_type("image/png", ["image/*"])

    def test_parameters_on_the_content_type_are_not_part_of_the_match(self) -> None:
        # A client that declares a charset sends the same media type.
        assert validate_content_type("text/plain; charset=utf-8", ["text/plain"])
        assert validate_content_type("text/plain;charset=utf-8", ["text/plain"])

    def test_parameters_on_the_allowed_mode_are_not_part_of_the_match(self) -> None:
        assert validate_content_type(
            "application/json", ["application/json; charset=utf-8"]
        )

    def test_parameters_do_not_mask_a_real_mismatch(self) -> None:
        assert not validate_content_type(
            "text/plain; charset=utf-8", ["application/json"]
        )


class TestValidateMessageParts:
    def test_file_part_with_charset_parameter_passes_validation(self) -> None:
        parts = [
            Part(root=TextPart(text="hello")),
            Part(
                root=FilePart(
                    file=FileWithBytes(
                        bytes="aGVsbG8=", mimeType="text/plain; charset=utf-8"
                    )
                )
            ),
        ]

        invalid = validate_message_parts(parts, ["text/plain", "application/json"])

        assert invalid == []

    def test_disallowed_file_part_is_still_reported(self) -> None:
        parts = [
            Part(
                root=FilePart(
                    file=FileWithBytes(bytes="aGVsbG8=", mimeType="image/png")
                )
            )
        ]

        invalid = validate_message_parts(parts, ["text/plain", "application/json"])

        assert invalid == ["image/png"]

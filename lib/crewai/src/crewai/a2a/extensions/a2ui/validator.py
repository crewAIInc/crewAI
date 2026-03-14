"""Validate A2UI message dicts via Pydantic models."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from crewai.a2a.extensions.a2ui.models import A2UIEvent, A2UIMessage


class A2UIValidationError(Exception):
    """Raised when an A2UI message fails validation."""

    def __init__(self, message: str, errors: list[Any] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def validate_a2ui_message(data: dict[str, Any]) -> A2UIMessage:
    """Parse and validate an A2UI server-to-client message.

    Args:
        data: Raw message dict (JSON-decoded).

    Returns:
        Validated ``A2UIMessage`` instance.

    Raises:
        A2UIValidationError: If the data does not conform to the A2UI schema.
    """
    try:
        return A2UIMessage.model_validate(data)
    except ValidationError as exc:
        raise A2UIValidationError(
            f"Invalid A2UI message: {exc.error_count()} validation error(s)",
            errors=exc.errors(),
        ) from exc


def validate_a2ui_event(data: dict[str, Any]) -> A2UIEvent:
    """Parse and validate an A2UI client-to-server event.

    Args:
        data: Raw event dict (JSON-decoded).

    Returns:
        Validated ``A2UIEvent`` instance.

    Raises:
        A2UIValidationError: If the data does not conform to the A2UI event schema.
    """
    try:
        return A2UIEvent.model_validate(data)
    except ValidationError as exc:
        raise A2UIValidationError(
            f"Invalid A2UI event: {exc.error_count()} validation error(s)",
            errors=exc.errors(),
        ) from exc

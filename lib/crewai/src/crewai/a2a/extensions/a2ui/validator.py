"""Validate A2UI message dicts via Pydantic models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from crewai.a2a.extensions.a2ui.catalog import (
    AudioPlayer,
    Button,
    Card,
    CheckBox,
    Column,
    DateTimeInput,
    Divider,
    Icon,
    Image,
    List,
    Modal,
    MultipleChoice,
    Row,
    Slider,
    Tabs,
    Text,
    TextField,
    Video,
)
from crewai.a2a.extensions.a2ui.models import A2UIEvent, A2UIMessage
from crewai.a2a.extensions.a2ui.v0_9 import (
    A2UIEventV09,
    A2UIMessageV09,
    AudioPlayerV09,
    ButtonV09,
    CardV09,
    CheckBoxV09,
    ChoicePickerV09,
    ColumnV09,
    DateTimeInputV09,
    DividerV09,
    IconV09,
    ImageV09,
    ListV09,
    ModalV09,
    RowV09,
    SliderV09,
    TabsV09,
    TextFieldV09,
    TextV09,
    VideoV09,
)


_STANDARD_CATALOG_MODELS: dict[str, type[BaseModel]] = {
    "AudioPlayer": AudioPlayer,
    "Button": Button,
    "Card": Card,
    "CheckBox": CheckBox,
    "Column": Column,
    "DateTimeInput": DateTimeInput,
    "Divider": Divider,
    "Icon": Icon,
    "Image": Image,
    "List": List,
    "Modal": Modal,
    "MultipleChoice": MultipleChoice,
    "Row": Row,
    "Slider": Slider,
    "Tabs": Tabs,
    "Text": Text,
    "TextField": TextField,
    "Video": Video,
}


class A2UIValidationError(Exception):
    """Raised when an A2UI message fails validation."""

    def __init__(self, message: str, errors: list[Any] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def validate_a2ui_message(
    data: dict[str, Any],
    *,
    validate_catalog: bool = False,
) -> A2UIMessage:
    """Parse and validate an A2UI server-to-client message.

    Args:
        data: Raw JSON-decoded message dict.
        validate_catalog: If True, also validate component properties
            against the standard catalog.

    Returns:
        Validated ``A2UIMessage`` instance.

    Raises:
        A2UIValidationError: If the data does not conform to the A2UI schema.
    """
    try:
        message = A2UIMessage.model_validate(data)
    except ValidationError as exc:
        raise A2UIValidationError(
            f"Invalid A2UI message: {exc.error_count()} validation error(s)",
            errors=exc.errors(),
        ) from exc

    if validate_catalog:
        validate_catalog_components(message)

    return message


def validate_a2ui_event(data: dict[str, Any]) -> A2UIEvent:
    """Parse and validate an A2UI client-to-server event.

    Args:
        data: Raw JSON-decoded event dict.

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


def validate_a2ui_message_v09(data: dict[str, Any]) -> A2UIMessageV09:
    """Parse and validate an A2UI v0.9 server-to-client message.

    Args:
        data: Raw JSON-decoded message dict.

    Returns:
        Validated ``A2UIMessageV09`` instance.

    Raises:
        A2UIValidationError: If the data does not conform to the v0.9 schema.
    """
    try:
        return A2UIMessageV09.model_validate(data)
    except ValidationError as exc:
        raise A2UIValidationError(
            f"Invalid A2UI v0.9 message: {exc.error_count()} validation error(s)",
            errors=exc.errors(),
        ) from exc


def validate_a2ui_event_v09(data: dict[str, Any]) -> A2UIEventV09:
    """Parse and validate an A2UI v0.9 client-to-server event.

    Args:
        data: Raw JSON-decoded event dict.

    Returns:
        Validated ``A2UIEventV09`` instance.

    Raises:
        A2UIValidationError: If the data does not conform to the v0.9 schema.
    """
    try:
        return A2UIEventV09.model_validate(data)
    except ValidationError as exc:
        raise A2UIValidationError(
            f"Invalid A2UI v0.9 event: {exc.error_count()} validation error(s)",
            errors=exc.errors(),
        ) from exc


def validate_catalog_components(message: A2UIMessage) -> None:
    """Validate component properties in a surfaceUpdate against the standard catalog.

    Only applies to surfaceUpdate messages. Components whose type is not
    in the standard catalog are skipped without error.

    Args:
        message: A validated A2UIMessage.

    Raises:
        A2UIValidationError: If any component fails catalog validation.
    """
    if message.surface_update is None:
        return

    errors: list[Any] = []
    for entry in message.surface_update.components:
        for type_name, props in entry.component.items():
            model = _STANDARD_CATALOG_MODELS.get(type_name)
            if model is None:
                continue
            try:
                model.model_validate(props)
            except ValidationError as exc:
                errors.extend(
                    {
                        "component_id": entry.id,
                        "component_type": type_name,
                        **err,
                    }
                    for err in exc.errors()
                )

    if errors:
        raise A2UIValidationError(
            f"Catalog validation failed: {len(errors)} error(s)",
            errors=errors,
        )


_V09_BASIC_CATALOG_MODELS: dict[str, type[BaseModel]] = {
    "AudioPlayer": AudioPlayerV09,
    "Button": ButtonV09,
    "Card": CardV09,
    "CheckBox": CheckBoxV09,
    "ChoicePicker": ChoicePickerV09,
    "Column": ColumnV09,
    "DateTimeInput": DateTimeInputV09,
    "Divider": DividerV09,
    "Icon": IconV09,
    "Image": ImageV09,
    "List": ListV09,
    "Modal": ModalV09,
    "Row": RowV09,
    "Slider": SliderV09,
    "Tabs": TabsV09,
    "Text": TextV09,
    "TextField": TextFieldV09,
    "Video": VideoV09,
}


def validate_catalog_components_v09(message: A2UIMessageV09) -> None:
    """Validate component properties in an updateComponents against the basic catalog.

    v0.9 components use a flat structure where ``component`` is a type-name
    string and properties sit at the top level of the component dict.

    Only applies to updateComponents messages. Components whose type is not
    in the basic catalog are skipped without error.

    Args:
        message: A validated A2UIMessageV09.

    Raises:
        A2UIValidationError: If any component fails catalog validation.
    """
    if message.update_components is None:
        return

    errors: list[Any] = []
    for entry in message.update_components.components:
        if not isinstance(entry, dict):
            continue
        type_name = entry.get("component")
        if not isinstance(type_name, str):
            continue
        model = _V09_BASIC_CATALOG_MODELS.get(type_name)
        if model is None:
            continue
        try:
            model.model_validate(entry)
        except ValidationError as exc:
            errors.extend(
                {
                    "component_id": entry.get("id", "<unknown>"),
                    "component_type": type_name,
                    **err,
                }
                for err in exc.errors()
            )

    if errors:
        raise A2UIValidationError(
            f"v0.9 catalog validation failed: {len(errors)} error(s)",
            errors=errors,
        )

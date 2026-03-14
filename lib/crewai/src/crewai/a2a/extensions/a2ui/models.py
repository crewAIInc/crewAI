"""Pydantic models for A2UI server-to-client messages and client-to-server events."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BoundValue(BaseModel):
    """A value that can be a literal or a data-model path reference."""

    literal_string: str | None = Field(None, alias="literalString")
    literal_number: float | None = Field(None, alias="literalNumber")
    literal_boolean: bool | None = Field(None, alias="literalBoolean")
    literal_array: list[str] | None = Field(None, alias="literalArray")
    path: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class MapEntry(BaseModel):
    """A single entry in a valueMap adjacency list, supporting recursive nesting."""

    key: str
    value_string: str | None = Field(None, alias="valueString")
    value_number: float | None = Field(None, alias="valueNumber")
    value_boolean: bool | None = Field(None, alias="valueBoolean")
    value_map: list[MapEntry] | None = Field(None, alias="valueMap")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class DataEntry(BaseModel):
    """A data model entry with a key and exactly one typed value."""

    key: str
    value_string: str | None = Field(None, alias="valueString")
    value_number: float | None = Field(None, alias="valueNumber")
    value_boolean: bool | None = Field(None, alias="valueBoolean")
    value_map: list[MapEntry] | None = Field(None, alias="valueMap")

    model_config = {"populate_by_name": True, "extra": "forbid"}


_HEX_COLOR_PATTERN: re.Pattern[str] = re.compile(r"^#[0-9a-fA-F]{6}$")


class Styles(BaseModel):
    """Surface styling information."""

    font: str | None = None
    primary_color: str | None = Field(
        None, alias="primaryColor", pattern=_HEX_COLOR_PATTERN.pattern
    )

    model_config = {"populate_by_name": True, "extra": "allow"}


class ComponentEntry(BaseModel):
    """A single component in a UI widget tree.

    The ``component`` dict must contain exactly one key — the component type
    name (e.g. ``"Text"``, ``"Button"``) — whose value holds the component
    properties.  Component internals are left as ``dict[str, Any]`` because
    they are catalog-dependent; use the typed helpers in ``catalog.py`` for
    the standard catalog.
    """

    id: str
    weight: float | None = None
    component: dict[str, Any]

    model_config = {"extra": "forbid"}


class BeginRendering(BaseModel):
    """Signals the client to begin rendering a surface."""

    surface_id: str = Field(alias="surfaceId")
    root: str
    catalog_id: str | None = Field(None, alias="catalogId")
    styles: Styles | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class SurfaceUpdate(BaseModel):
    """Updates a surface with a new set of components."""

    surface_id: str = Field(alias="surfaceId")
    components: list[ComponentEntry] = Field(min_length=1)

    model_config = {"populate_by_name": True, "extra": "forbid"}


class DataModelUpdate(BaseModel):
    """Updates the data model for a surface."""

    surface_id: str = Field(alias="surfaceId")
    path: str | None = None
    contents: list[DataEntry]

    model_config = {"populate_by_name": True, "extra": "forbid"}


class DeleteSurface(BaseModel):
    """Signals the client to delete a surface."""

    surface_id: str = Field(alias="surfaceId")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class A2UIMessage(BaseModel):
    """Union wrapper for the four server-to-client A2UI message types.

    Exactly one of the fields must be set.
    """

    begin_rendering: BeginRendering | None = Field(None, alias="beginRendering")
    surface_update: SurfaceUpdate | None = Field(None, alias="surfaceUpdate")
    data_model_update: DataModelUpdate | None = Field(None, alias="dataModelUpdate")
    delete_surface: DeleteSurface | None = Field(None, alias="deleteSurface")

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="after")
    def _check_exactly_one(self) -> A2UIMessage:
        """Enforce the spec's exactly-one-of constraint."""
        fields = [
            self.begin_rendering,
            self.surface_update,
            self.data_model_update,
            self.delete_surface,
        ]
        count = sum(f is not None for f in fields)
        if count != 1:
            raise ValueError(f"Exactly one A2UI message type must be set, got {count}")
        return self


class UserAction(BaseModel):
    """Reports a user-initiated action from a component."""

    name: str
    surface_id: str = Field(alias="surfaceId")
    source_component_id: str = Field(alias="sourceComponentId")
    timestamp: str
    context: dict[str, Any]

    model_config = {"populate_by_name": True}


class ClientError(BaseModel):
    """Reports a client-side error."""

    model_config = {"extra": "allow"}


class A2UIEvent(BaseModel):
    """Union wrapper for client-to-server events."""

    user_action: UserAction | None = Field(None, alias="userAction")
    error: ClientError | None = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _check_exactly_one(self) -> A2UIEvent:
        """Enforce the spec's exactly-one-of constraint."""
        fields = [self.user_action, self.error]
        count = sum(f is not None for f in fields)
        if count != 1:
            raise ValueError(f"Exactly one A2UI event type must be set, got {count}")
        return self


class A2UIResponse(BaseModel):
    """Typed wrapper for responses containing A2UI messages."""

    text: str
    a2ui_parts: list[dict[str, Any]] = Field(default_factory=list)
    a2ui_messages: list[dict[str, Any]] = Field(default_factory=list)


_A2UI_KEYS = {"beginRendering", "surfaceUpdate", "dataModelUpdate", "deleteSurface"}


def extract_a2ui_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects containing A2UI keys from text.

    Uses ``json.JSONDecoder.raw_decode`` for robust parsing that correctly
    handles braces inside string literals.
    """
    decoder = json.JSONDecoder()
    results: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        idx = text.find("{", idx)
        if idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            if isinstance(obj, dict) and _A2UI_KEYS & obj.keys():
                results.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1
    return results

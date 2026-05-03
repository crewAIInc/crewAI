"""Pydantic models for A2UI server-to-client messages and client-to-server events."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundValue(BaseModel):
    """A value that can be a literal or a data-model path reference."""

    literal_string: str | None = Field(
        default=None, alias="literalString", description="Literal string value."
    )
    literal_number: float | None = Field(
        default=None, alias="literalNumber", description="Literal numeric value."
    )
    literal_boolean: bool | None = Field(
        default=None, alias="literalBoolean", description="Literal boolean value."
    )
    literal_array: list[str] | None = Field(
        default=None, alias="literalArray", description="Literal array of strings."
    )
    path: str | None = Field(default=None, description="Data-model path reference.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class MapEntry(BaseModel):
    """A single entry in a valueMap adjacency list, supporting recursive nesting."""

    key: str = Field(description="Entry key.")
    value_string: str | None = Field(
        default=None, alias="valueString", description="String value."
    )
    value_number: float | None = Field(
        default=None, alias="valueNumber", description="Numeric value."
    )
    value_boolean: bool | None = Field(
        default=None, alias="valueBoolean", description="Boolean value."
    )
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DataEntry(BaseModel):
    """A data model entry with a key and exactly one typed value."""

    key: str = Field(description="Entry key.")
    value_string: str | None = Field(
        default=None, alias="valueString", description="String value."
    )
    value_number: float | None = Field(
        default=None, alias="valueNumber", description="Numeric value."
    )
    value_boolean: bool | None = Field(
        default=None, alias="valueBoolean", description="Boolean value."
    )
    value_map: list[MapEntry] | None = Field(
        default=None, alias="valueMap", description="Nested map entries."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


_HEX_COLOR_PATTERN: re.Pattern[str] = re.compile(r"^#[0-9a-fA-F]{6}$")


class Styles(BaseModel):
    """Surface styling information."""

    font: str | None = Field(default=None, description="Font family name.")
    primary_color: str | None = Field(
        default=None,
        alias="primaryColor",
        pattern=_HEX_COLOR_PATTERN.pattern,
        description="Primary color as a hex string.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ComponentEntry(BaseModel):
    """A single component in a UI widget tree.

    The ``component`` dict must contain exactly one key — the component type
    name (e.g. ``"Text"``, ``"Button"``) — whose value holds the component
    properties.  Component internals are left as ``dict[str, Any]`` because
    they are catalog-dependent; use the typed helpers in ``catalog.py`` for
    the standard catalog.
    """

    id: str = Field(description="Unique component identifier.")
    weight: float | None = Field(
        default=None, description="Flex weight for layout distribution."
    )
    component: dict[str, Any] = Field(
        description="Component type name mapped to its properties."
    )

    model_config = ConfigDict(extra="forbid")


class BeginRendering(BaseModel):
    """Signals the client to begin rendering a surface."""

    surface_id: str = Field(alias="surfaceId", description="Unique surface identifier.")
    root: str = Field(description="Component ID of the root element.")
    catalog_id: str | None = Field(
        default=None,
        alias="catalogId",
        description="Catalog identifier for the surface.",
    )
    styles: Styles | None = Field(
        default=None, description="Surface styling overrides."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class SurfaceUpdate(BaseModel):
    """Updates a surface with a new set of components."""

    surface_id: str = Field(alias="surfaceId", description="Target surface identifier.")
    components: list[ComponentEntry] = Field(
        min_length=1, description="Components to render on the surface."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DataModelUpdate(BaseModel):
    """Updates the data model for a surface."""

    surface_id: str = Field(alias="surfaceId", description="Target surface identifier.")
    path: str | None = Field(
        default=None, description="Data-model path prefix for the update."
    )
    contents: list[DataEntry] = Field(
        description="Data entries to merge into the model."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DeleteSurface(BaseModel):
    """Signals the client to delete a surface."""

    surface_id: str = Field(
        alias="surfaceId", description="Surface identifier to delete."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class A2UIMessage(BaseModel):
    """Union wrapper for the four server-to-client A2UI message types.

    Exactly one of the fields must be set.
    """

    begin_rendering: BeginRendering | None = Field(
        default=None,
        alias="beginRendering",
        description="Begin rendering a new surface.",
    )
    surface_update: SurfaceUpdate | None = Field(
        default=None,
        alias="surfaceUpdate",
        description="Update components on a surface.",
    )
    data_model_update: DataModelUpdate | None = Field(
        default=None,
        alias="dataModelUpdate",
        description="Update the surface data model.",
    )
    delete_surface: DeleteSurface | None = Field(
        default=None, alias="deleteSurface", description="Delete an existing surface."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

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

    name: str = Field(description="Action name.")
    surface_id: str = Field(alias="surfaceId", description="Source surface identifier.")
    source_component_id: str = Field(
        alias="sourceComponentId", description="Component that triggered the action."
    )
    timestamp: str = Field(description="ISO 8601 timestamp of the action.")
    context: dict[str, Any] = Field(description="Action context payload.")

    model_config = ConfigDict(populate_by_name=True)


class ClientError(BaseModel):
    """Reports a client-side error."""

    model_config = ConfigDict(extra="allow")


class A2UIEvent(BaseModel):
    """Union wrapper for client-to-server events."""

    user_action: UserAction | None = Field(
        default=None, alias="userAction", description="User-initiated action event."
    )
    error: ClientError | None = Field(
        default=None, description="Client-side error report."
    )

    model_config = ConfigDict(populate_by_name=True)

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

    text: str = Field(description="Raw text content of the response.")
    a2ui_parts: list[dict[str, Any]] = Field(
        default_factory=list, description="A2UI DataParts extracted from the response."
    )
    a2ui_messages: list[dict[str, Any]] = Field(
        default_factory=list, description="Validated A2UI message dicts."
    )


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

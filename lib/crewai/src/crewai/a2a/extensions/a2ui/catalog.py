"""Typed helpers for A2UI standard catalog components.

These models provide optional type safety for standard catalog components.
Agents can also use raw dicts validated against the JSON schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StringBinding(BaseModel):
    """A string value: literal or data-model path."""

    literal_string: str | None = Field(None, alias="literalString")
    path: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class NumberBinding(BaseModel):
    """A numeric value: literal or data-model path."""

    literal_number: float | None = Field(None, alias="literalNumber")
    path: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class BooleanBinding(BaseModel):
    """A boolean value: literal or data-model path."""

    literal_boolean: bool | None = Field(None, alias="literalBoolean")
    path: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ArrayBinding(BaseModel):
    """An array value: literal or data-model path."""

    literal_array: list[str] | None = Field(None, alias="literalArray")
    path: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ChildrenDef(BaseModel):
    """Children definition for layout components."""

    explicit_list: list[str] | None = Field(None, alias="explicitList")
    template: ChildTemplate | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ChildTemplate(BaseModel):
    """Template for generating dynamic children from a data model list."""

    component_id: str = Field(alias="componentId")
    data_binding: str = Field(alias="dataBinding")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class ActionContextEntry(BaseModel):
    """A key-value pair in an action context payload."""

    key: str
    value: ActionBoundValue

    model_config = {"extra": "forbid"}


class ActionBoundValue(BaseModel):
    """A value in an action context: literal or data-model path."""

    path: str | None = None
    literal_string: str | None = Field(None, alias="literalString")
    literal_number: float | None = Field(None, alias="literalNumber")
    literal_boolean: bool | None = Field(None, alias="literalBoolean")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Action(BaseModel):
    """Client-side action dispatched by interactive components."""

    name: str
    context: list[ActionContextEntry] | None = None

    model_config = {"extra": "forbid"}


class TabItem(BaseModel):
    """A single tab definition."""

    title: StringBinding
    child: str

    model_config = {"extra": "forbid"}


class MultipleChoiceOption(BaseModel):
    """A single option in a MultipleChoice component."""

    label: StringBinding
    value: str

    model_config = {"extra": "forbid"}


class Text(BaseModel):
    """Displays text content."""

    text: StringBinding
    usage_hint: Literal["h1", "h2", "h3", "h4", "h5", "caption", "body"] | None = Field(
        None, alias="usageHint"
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Image(BaseModel):
    """Displays an image."""

    url: StringBinding
    fit: Literal["contain", "cover", "fill", "none", "scale-down"] | None = None
    usage_hint: (
        Literal[
            "icon", "avatar", "smallFeature", "mediumFeature", "largeFeature", "header"
        ]
        | None
    ) = Field(None, alias="usageHint")

    model_config = {"populate_by_name": True, "extra": "forbid"}


IconName = Literal[
    "accountCircle",
    "add",
    "arrowBack",
    "arrowForward",
    "attachFile",
    "calendarToday",
    "call",
    "camera",
    "check",
    "close",
    "delete",
    "download",
    "edit",
    "event",
    "error",
    "favorite",
    "favoriteOff",
    "folder",
    "help",
    "home",
    "info",
    "locationOn",
    "lock",
    "lockOpen",
    "mail",
    "menu",
    "moreVert",
    "moreHoriz",
    "notificationsOff",
    "notifications",
    "payment",
    "person",
    "phone",
    "photo",
    "print",
    "refresh",
    "search",
    "send",
    "settings",
    "share",
    "shoppingCart",
    "star",
    "starHalf",
    "starOff",
    "upload",
    "visibility",
    "visibilityOff",
    "warning",
]


class IconBinding(BaseModel):
    """Icon name: literal enum or data-model path."""

    literal_string: IconName | None = Field(None, alias="literalString")
    path: str | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Icon(BaseModel):
    """Displays a named icon."""

    name: IconBinding

    model_config = {"extra": "forbid"}


class Video(BaseModel):
    """Displays a video player."""

    url: StringBinding

    model_config = {"extra": "forbid"}


class AudioPlayer(BaseModel):
    """Displays an audio player."""

    url: StringBinding
    description: StringBinding | None = None

    model_config = {"extra": "forbid"}


class Row(BaseModel):
    """Horizontal layout container."""

    children: ChildrenDef
    distribution: (
        Literal["center", "end", "spaceAround", "spaceBetween", "spaceEvenly", "start"]
        | None
    ) = None
    alignment: Literal["start", "center", "end", "stretch"] | None = None

    model_config = {"extra": "forbid"}


class Column(BaseModel):
    """Vertical layout container."""

    children: ChildrenDef
    distribution: (
        Literal["start", "center", "end", "spaceBetween", "spaceAround", "spaceEvenly"]
        | None
    ) = None
    alignment: Literal["center", "end", "start", "stretch"] | None = None

    model_config = {"extra": "forbid"}


class List(BaseModel):
    """Scrollable list container."""

    children: ChildrenDef
    direction: Literal["vertical", "horizontal"] | None = None
    alignment: Literal["start", "center", "end", "stretch"] | None = None

    model_config = {"extra": "forbid"}


class Card(BaseModel):
    """Card container wrapping a single child."""

    child: str

    model_config = {"extra": "forbid"}


class Tabs(BaseModel):
    """Tabbed navigation container."""

    tab_items: list[TabItem] = Field(alias="tabItems")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Divider(BaseModel):
    """A visual divider line."""

    axis: Literal["horizontal", "vertical"] | None = None

    model_config = {"extra": "forbid"}


class Modal(BaseModel):
    """A modal dialog with an entry point trigger and content."""

    entry_point_child: str = Field(alias="entryPointChild")
    content_child: str = Field(alias="contentChild")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Button(BaseModel):
    """An interactive button with an action."""

    child: str
    primary: bool | None = None
    action: Action

    model_config = {"extra": "forbid"}


class CheckBox(BaseModel):
    """A checkbox input."""

    label: StringBinding
    value: BooleanBinding

    model_config = {"extra": "forbid"}


class TextField(BaseModel):
    """A text input field."""

    label: StringBinding
    text: StringBinding | None = None
    text_field_type: (
        Literal["date", "longText", "number", "shortText", "obscured"] | None
    ) = Field(None, alias="textFieldType")
    validation_regexp: str | None = Field(None, alias="validationRegexp")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class DateTimeInput(BaseModel):
    """A date and/or time picker."""

    value: StringBinding
    enable_date: bool | None = Field(None, alias="enableDate")
    enable_time: bool | None = Field(None, alias="enableTime")

    model_config = {"populate_by_name": True, "extra": "forbid"}


class MultipleChoice(BaseModel):
    """A multiple-choice selection component."""

    selections: ArrayBinding
    options: list[MultipleChoiceOption]
    max_allowed_selections: int | None = Field(None, alias="maxAllowedSelections")
    variant: Literal["checkbox", "chips"] | None = None
    filterable: bool | None = None

    model_config = {"populate_by_name": True, "extra": "forbid"}


class Slider(BaseModel):
    """A numeric slider input."""

    value: NumberBinding
    min_value: float | None = Field(None, alias="minValue")
    max_value: float | None = Field(None, alias="maxValue")

    model_config = {"populate_by_name": True, "extra": "forbid"}


STANDARD_CATALOG_COMPONENTS: frozenset[str] = frozenset(
    {
        "Text",
        "Image",
        "Icon",
        "Video",
        "AudioPlayer",
        "Row",
        "Column",
        "List",
        "Card",
        "Tabs",
        "Divider",
        "Modal",
        "Button",
        "CheckBox",
        "TextField",
        "DateTimeInput",
        "MultipleChoice",
        "Slider",
    }
)

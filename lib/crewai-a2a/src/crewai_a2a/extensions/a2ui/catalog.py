"""Typed helpers for A2UI standard catalog components.

These models provide optional type safety for standard catalog components.
Agents can also use raw dicts validated against the JSON schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StringBinding(BaseModel):
    """A string value: literal or data-model path."""

    literal_string: str | None = Field(
        default=None, alias="literalString", description="Literal string value."
    )
    path: str | None = Field(default=None, description="Data-model path reference.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class NumberBinding(BaseModel):
    """A numeric value: literal or data-model path."""

    literal_number: float | None = Field(
        default=None, alias="literalNumber", description="Literal numeric value."
    )
    path: str | None = Field(default=None, description="Data-model path reference.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class BooleanBinding(BaseModel):
    """A boolean value: literal or data-model path."""

    literal_boolean: bool | None = Field(
        default=None, alias="literalBoolean", description="Literal boolean value."
    )
    path: str | None = Field(default=None, description="Data-model path reference.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ArrayBinding(BaseModel):
    """An array value: literal or data-model path."""

    literal_array: list[str] | None = Field(
        default=None, alias="literalArray", description="Literal array of strings."
    )
    path: str | None = Field(default=None, description="Data-model path reference.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ChildrenDef(BaseModel):
    """Children definition for layout components."""

    explicit_list: list[str] | None = Field(
        default=None,
        alias="explicitList",
        description="Explicit list of child component IDs.",
    )
    template: ChildTemplate | None = Field(
        default=None, description="Template for generating dynamic children."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ChildTemplate(BaseModel):
    """Template for generating dynamic children from a data model list."""

    component_id: str = Field(
        alias="componentId", description="ID of the component to repeat."
    )
    data_binding: str = Field(
        alias="dataBinding", description="Data-model path to bind the template to."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ActionContextEntry(BaseModel):
    """A key-value pair in an action context payload."""

    key: str = Field(description="Context entry key.")
    value: ActionBoundValue = Field(description="Context entry value.")

    model_config = ConfigDict(extra="forbid")


class ActionBoundValue(BaseModel):
    """A value in an action context: literal or data-model path."""

    path: str | None = Field(default=None, description="Data-model path reference.")
    literal_string: str | None = Field(
        default=None, alias="literalString", description="Literal string value."
    )
    literal_number: float | None = Field(
        default=None, alias="literalNumber", description="Literal numeric value."
    )
    literal_boolean: bool | None = Field(
        default=None, alias="literalBoolean", description="Literal boolean value."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Action(BaseModel):
    """Client-side action dispatched by interactive components."""

    name: str = Field(description="Action name dispatched on interaction.")
    context: list[ActionContextEntry] | None = Field(
        default=None, description="Key-value pairs sent with the action."
    )

    model_config = ConfigDict(extra="forbid")


class TabItem(BaseModel):
    """A single tab definition."""

    title: StringBinding = Field(description="Tab title text.")
    child: str = Field(description="Component ID rendered as the tab content.")

    model_config = ConfigDict(extra="forbid")


class MultipleChoiceOption(BaseModel):
    """A single option in a MultipleChoice component."""

    label: StringBinding = Field(description="Display label for the option.")
    value: str = Field(description="Value submitted when the option is selected.")

    model_config = ConfigDict(extra="forbid")


class Text(BaseModel):
    """Displays text content."""

    text: StringBinding = Field(description="Text content to display.")
    usage_hint: Literal["h1", "h2", "h3", "h4", "h5", "caption", "body"] | None = Field(
        default=None, alias="usageHint", description="Semantic hint for text styling."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Image(BaseModel):
    """Displays an image."""

    url: StringBinding = Field(description="Image source URL.")
    fit: Literal["contain", "cover", "fill", "none", "scale-down"] | None = Field(
        default=None, description="Object-fit behavior for the image."
    )
    usage_hint: (
        Literal[
            "icon", "avatar", "smallFeature", "mediumFeature", "largeFeature", "header"
        ]
        | None
    ) = Field(
        default=None, alias="usageHint", description="Semantic hint for image sizing."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


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

    literal_string: IconName | None = Field(
        default=None, alias="literalString", description="Literal icon name."
    )
    path: str | None = Field(default=None, description="Data-model path reference.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Icon(BaseModel):
    """Displays a named icon."""

    name: IconBinding = Field(description="Icon name binding.")

    model_config = ConfigDict(extra="forbid")


class Video(BaseModel):
    """Displays a video player."""

    url: StringBinding = Field(description="Video source URL.")

    model_config = ConfigDict(extra="forbid")


class AudioPlayer(BaseModel):
    """Displays an audio player."""

    url: StringBinding = Field(description="Audio source URL.")
    description: StringBinding | None = Field(
        default=None, description="Accessible description of the audio content."
    )

    model_config = ConfigDict(extra="forbid")


class Row(BaseModel):
    """Horizontal layout container."""

    children: ChildrenDef = Field(description="Child components in this row.")
    distribution: (
        Literal["center", "end", "spaceAround", "spaceBetween", "spaceEvenly", "start"]
        | None
    ) = Field(
        default=None, description="How children are distributed along the main axis."
    )
    alignment: Literal["start", "center", "end", "stretch"] | None = Field(
        default=None, description="How children are aligned on the cross axis."
    )

    model_config = ConfigDict(extra="forbid")


class Column(BaseModel):
    """Vertical layout container."""

    children: ChildrenDef = Field(description="Child components in this column.")
    distribution: (
        Literal["start", "center", "end", "spaceBetween", "spaceAround", "spaceEvenly"]
        | None
    ) = Field(
        default=None, description="How children are distributed along the main axis."
    )
    alignment: Literal["center", "end", "start", "stretch"] | None = Field(
        default=None, description="How children are aligned on the cross axis."
    )

    model_config = ConfigDict(extra="forbid")


class List(BaseModel):
    """Scrollable list container."""

    children: ChildrenDef = Field(description="Child components in this list.")
    direction: Literal["vertical", "horizontal"] | None = Field(
        default=None, description="Scroll direction of the list."
    )
    alignment: Literal["start", "center", "end", "stretch"] | None = Field(
        default=None, description="How children are aligned on the cross axis."
    )

    model_config = ConfigDict(extra="forbid")


class Card(BaseModel):
    """Card container wrapping a single child."""

    child: str = Field(description="Component ID of the card content.")

    model_config = ConfigDict(extra="forbid")


class Tabs(BaseModel):
    """Tabbed navigation container."""

    tab_items: list[TabItem] = Field(
        alias="tabItems", description="List of tab definitions."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Divider(BaseModel):
    """A visual divider line."""

    axis: Literal["horizontal", "vertical"] | None = Field(
        default=None, description="Orientation of the divider."
    )

    model_config = ConfigDict(extra="forbid")


class Modal(BaseModel):
    """A modal dialog with an entry point trigger and content."""

    entry_point_child: str = Field(
        alias="entryPointChild", description="Component ID that triggers the modal."
    )
    content_child: str = Field(
        alias="contentChild", description="Component ID rendered inside the modal."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Button(BaseModel):
    """An interactive button with an action."""

    child: str = Field(description="Component ID of the button label.")
    primary: bool | None = Field(
        default=None, description="Whether the button uses primary styling."
    )
    action: Action = Field(description="Action dispatched when the button is clicked.")

    model_config = ConfigDict(extra="forbid")


class CheckBox(BaseModel):
    """A checkbox input."""

    label: StringBinding = Field(description="Label text for the checkbox.")
    value: BooleanBinding = Field(
        description="Boolean value binding for the checkbox state."
    )

    model_config = ConfigDict(extra="forbid")


class TextField(BaseModel):
    """A text input field."""

    label: StringBinding = Field(description="Label text for the input.")
    text: StringBinding | None = Field(
        default=None, description="Current text value binding."
    )
    text_field_type: (
        Literal["date", "longText", "number", "shortText", "obscured"] | None
    ) = Field(default=None, alias="textFieldType", description="Input type variant.")
    validation_regexp: str | None = Field(
        default=None,
        alias="validationRegexp",
        description="Regex pattern for client-side validation.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DateTimeInput(BaseModel):
    """A date and/or time picker."""

    value: StringBinding = Field(description="ISO date/time string value binding.")
    enable_date: bool | None = Field(
        default=None,
        alias="enableDate",
        description="Whether the date picker is enabled.",
    )
    enable_time: bool | None = Field(
        default=None,
        alias="enableTime",
        description="Whether the time picker is enabled.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class MultipleChoice(BaseModel):
    """A multiple-choice selection component."""

    selections: ArrayBinding = Field(description="Array binding for selected values.")
    options: list[MultipleChoiceOption] = Field(description="Available choices.")
    max_allowed_selections: int | None = Field(
        default=None,
        alias="maxAllowedSelections",
        description="Maximum number of selections allowed.",
    )
    variant: Literal["checkbox", "chips"] | None = Field(
        default=None, description="Visual variant for the selection UI."
    )
    filterable: bool | None = Field(
        default=None, description="Whether options can be filtered by typing."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Slider(BaseModel):
    """A numeric slider input."""

    value: NumberBinding = Field(
        description="Numeric value binding for the slider position."
    )
    min_value: float | None = Field(
        default=None, alias="minValue", description="Minimum slider value."
    )
    max_value: float | None = Field(
        default=None, alias="maxValue", description="Maximum slider value."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


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

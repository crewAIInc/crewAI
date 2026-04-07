"""Pydantic models for A2UI v0.9 protocol messages and types.

This module provides v0.9 counterparts to the v0.8 models in ``models.py``.
Key differences from v0.8:

* ``beginRendering`` → ``createSurface`` — adds ``theme``, ``sendDataModel``,
  requires ``catalogId``.
* ``surfaceUpdate`` → ``updateComponents`` — component structure is flat:
  ``component`` is a type-name string, properties live at the top level.
* ``dataModelUpdate`` → ``updateDataModel`` — ``contents`` adjacency list
  replaced by a single ``value`` of any JSON type; ``path`` uses JSON Pointers.
* All messages carry a ``version: "v0.9"`` discriminator.
* Data binding uses plain JSON values, ``DataBinding`` objects, or
  ``FunctionCall`` objects instead of ``literalString`` / ``path`` wrappers.
* ``MultipleChoice`` is replaced by ``ChoicePicker``.
* ``Styles`` is replaced by ``Theme`` — adds ``iconUrl``, ``agentDisplayName``.
* Client-to-server ``userAction`` is renamed to ``action``; ``error`` gains
  structured ``code`` / ``path`` fields.
"""

from __future__ import annotations

import json
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


ComponentName = Literal[
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
    "Modal",
    "Divider",
    "Button",
    "TextField",
    "CheckBox",
    "ChoicePicker",
    "Slider",
    "DateTimeInput",
]

BASIC_CATALOG_COMPONENTS: frozenset[ComponentName] = frozenset(get_args(ComponentName))

FunctionName = Literal[
    "required",
    "regex",
    "length",
    "numeric",
    "email",
    "formatString",
    "formatNumber",
    "formatCurrency",
    "formatDate",
    "pluralize",
    "openUrl",
    "and",
    "or",
    "not",
]

BASIC_CATALOG_FUNCTIONS: frozenset[FunctionName] = frozenset(get_args(FunctionName))

IconNameV09 = Literal[
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
    "fastForward",
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
    "pause",
    "payment",
    "person",
    "phone",
    "photo",
    "play",
    "print",
    "refresh",
    "rewind",
    "search",
    "send",
    "settings",
    "share",
    "shoppingCart",
    "skipNext",
    "skipPrevious",
    "star",
    "starHalf",
    "starOff",
    "stop",
    "upload",
    "visibility",
    "visibilityOff",
    "volumeDown",
    "volumeMute",
    "volumeOff",
    "volumeUp",
    "warning",
]

V09_ICON_NAMES: frozenset[IconNameV09] = frozenset(get_args(IconNameV09))


class DataBinding(BaseModel):
    """JSON Pointer path reference to the data model."""

    path: str = Field(description="A JSON Pointer path to a value in the data model.")

    model_config = ConfigDict(extra="forbid")


class FunctionCall(BaseModel):
    """Client-side function invocation."""

    call: str = Field(description="The name of the function to call.")
    args: dict[str, DynamicValue] | None = Field(
        default=None, description="Arguments passed to the function."
    )
    return_type: (
        Literal["string", "number", "boolean", "array", "object", "any", "void"] | None
    ) = Field(
        default=None,
        alias="returnType",
        description="Expected return type of the function call.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


DynamicValue = str | float | int | bool | list[Any] | DataBinding | FunctionCall
DynamicString = str | DataBinding | FunctionCall
DynamicNumber = float | int | DataBinding | FunctionCall
DynamicBoolean = bool | DataBinding | FunctionCall
DynamicStringList = list[str] | DataBinding | FunctionCall


class CheckRule(BaseModel):
    """A single validation rule for an input component."""

    condition: DynamicBoolean = Field(
        description="Condition that must evaluate to true for the check to pass."
    )
    message: str = Field(description="Error message displayed if the check fails.")

    model_config = ConfigDict(extra="forbid")


class AccessibilityAttributes(BaseModel):
    """Accessibility attributes for assistive technologies."""

    label: DynamicString | None = Field(
        default=None, description="Short label for screen readers."
    )
    description: DynamicString | None = Field(
        default=None, description="Extended description for screen readers."
    )

    model_config = ConfigDict(extra="forbid")


class ChildTemplate(BaseModel):
    """Template for generating dynamic children from a data model list."""

    component_id: str = Field(
        alias="componentId", description="Component to repeat per list item."
    )
    path: str = Field(description="Data model path to the list of items.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


ChildListV09 = list[str] | ChildTemplate


class EventAction(BaseModel):
    """Server-side event triggered by a component interaction."""

    name: str = Field(description="Action name dispatched to the server.")
    context: dict[str, DynamicValue] | None = Field(
        default=None, description="Key-value pairs sent with the event."
    )

    model_config = ConfigDict(extra="forbid")


class ActionV09(BaseModel):
    """Interaction handler: server event or local function call.

    Exactly one of ``event`` or ``function_call`` must be set.
    """

    event: EventAction | None = Field(
        default=None, description="Triggers a server-side event."
    )
    function_call: FunctionCall | None = Field(
        default=None,
        alias="functionCall",
        description="Executes a local client-side function.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="after")
    def _check_exactly_one(self) -> ActionV09:
        """Enforce exactly one of event or functionCall."""
        count = sum(f is not None for f in (self.event, self.function_call))
        if count != 1:
            raise ValueError(
                f"Exactly one of event or functionCall must be set, got {count}"
            )
        return self


class TextV09(BaseModel):
    """Displays text content."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Text"] = "Text"
    text: DynamicString = Field(description="Text content to display.")
    variant: Literal["h1", "h2", "h3", "h4", "h5", "caption", "body"] | None = Field(
        default=None, description="Semantic text style hint."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class ImageV09(BaseModel):
    """Displays an image."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Image"] = "Image"
    url: DynamicString = Field(description="Image source URL.")
    description: DynamicString | None = Field(
        default=None, description="Accessibility text."
    )
    fit: Literal["contain", "cover", "fill", "none", "scaleDown"] | None = Field(
        default=None, description="Object-fit behavior."
    )
    variant: (
        Literal[
            "icon", "avatar", "smallFeature", "mediumFeature", "largeFeature", "header"
        ]
        | None
    ) = Field(default=None, description="Image size hint.")
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class IconV09(BaseModel):
    """Displays a named icon."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Icon"] = "Icon"
    name: IconNameV09 | DataBinding = Field(description="Icon name or data binding.")
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class VideoV09(BaseModel):
    """Displays a video player."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Video"] = "Video"
    url: DynamicString = Field(description="Video source URL.")
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class AudioPlayerV09(BaseModel):
    """Displays an audio player."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["AudioPlayer"] = "AudioPlayer"
    url: DynamicString = Field(description="Audio source URL.")
    description: DynamicString | None = Field(
        default=None, description="Audio content description."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class RowV09(BaseModel):
    """Horizontal layout container."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Row"] = "Row"
    children: ChildListV09 = Field(description="Child components.")
    justify: (
        Literal[
            "center",
            "end",
            "spaceAround",
            "spaceBetween",
            "spaceEvenly",
            "start",
            "stretch",
        ]
        | None
    ) = Field(default=None, description="Main-axis distribution.")
    align: Literal["start", "center", "end", "stretch"] | None = Field(
        default=None, description="Cross-axis alignment."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class ColumnV09(BaseModel):
    """Vertical layout container."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Column"] = "Column"
    children: ChildListV09 = Field(description="Child components.")
    justify: (
        Literal[
            "start",
            "center",
            "end",
            "spaceBetween",
            "spaceAround",
            "spaceEvenly",
            "stretch",
        ]
        | None
    ) = Field(default=None, description="Main-axis distribution.")
    align: Literal["center", "end", "start", "stretch"] | None = Field(
        default=None, description="Cross-axis alignment."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class ListV09(BaseModel):
    """Scrollable list container."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["List"] = "List"
    children: ChildListV09 = Field(description="Child components.")
    direction: Literal["vertical", "horizontal"] | None = Field(
        default=None, description="Scroll direction."
    )
    align: Literal["start", "center", "end", "stretch"] | None = Field(
        default=None, description="Cross-axis alignment."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class CardV09(BaseModel):
    """Card container wrapping a single child."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Card"] = "Card"
    child: str = Field(description="ID of the child component.")
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class TabItemV09(BaseModel):
    """A single tab definition."""

    title: DynamicString = Field(description="Tab title.")
    child: str = Field(description="ID of the tab content component.")

    model_config = ConfigDict(extra="forbid")


class TabsV09(BaseModel):
    """Tabbed navigation container."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Tabs"] = "Tabs"
    tabs: list[TabItemV09] = Field(min_length=1, description="Tab definitions.")
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class ModalV09(BaseModel):
    """Modal dialog with a trigger and content."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Modal"] = "Modal"
    trigger: str = Field(description="ID of the component that opens the modal.")
    content: str = Field(description="ID of the component inside the modal.")
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class DividerV09(BaseModel):
    """Visual divider line."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Divider"] = "Divider"
    axis: Literal["horizontal", "vertical"] | None = Field(
        default=None, description="Divider orientation."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class ButtonV09(BaseModel):
    """Interactive button."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Button"] = "Button"
    child: str = Field(description="ID of the button label component.")
    action: ActionV09 = Field(description="Action dispatched on click.")
    variant: Literal["default", "primary", "borderless"] | None = Field(
        default=None, description="Button style variant."
    )
    checks: list[CheckRule] | None = Field(
        default=None, description="Validation rules."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class TextFieldV09(BaseModel):
    """Text input field."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["TextField"] = "TextField"
    label: DynamicString = Field(description="Input label.")
    value: DynamicString | None = Field(default=None, description="Current text value.")
    variant: Literal["longText", "number", "shortText", "obscured"] | None = Field(
        default=None, description="Input type variant."
    )
    validation_regexp: str | None = Field(
        default=None,
        alias="validationRegexp",
        description="Regex for client-side validation.",
    )
    checks: list[CheckRule] | None = Field(
        default=None, description="Validation rules."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CheckBoxV09(BaseModel):
    """Checkbox input."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["CheckBox"] = "CheckBox"
    label: DynamicString = Field(description="Checkbox label.")
    value: DynamicBoolean = Field(description="Checked state.")
    checks: list[CheckRule] | None = Field(
        default=None, description="Validation rules."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class ChoicePickerOption(BaseModel):
    """A single option in a ChoicePicker."""

    label: DynamicString = Field(description="Display label.")
    value: str = Field(description="Value when selected.")

    model_config = ConfigDict(extra="forbid")


class ChoicePickerV09(BaseModel):
    """Selection component replacing v0.8 MultipleChoice."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["ChoicePicker"] = "ChoicePicker"
    options: list[ChoicePickerOption] = Field(description="Available choices.")
    value: DynamicStringList = Field(description="Currently selected values.")
    label: DynamicString | None = Field(default=None, description="Group label.")
    variant: Literal["multipleSelection", "mutuallyExclusive"] | None = Field(
        default=None, description="Selection behavior."
    )
    display_style: Literal["checkbox", "chips"] | None = Field(
        default=None, alias="displayStyle", description="Visual display style."
    )
    filterable: bool | None = Field(
        default=None, description="Whether options can be filtered."
    )
    checks: list[CheckRule] | None = Field(
        default=None, description="Validation rules."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class SliderV09(BaseModel):
    """Numeric slider input."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["Slider"] = "Slider"
    value: DynamicNumber = Field(description="Current slider value.")
    max: float = Field(description="Maximum slider value.")
    min: float | None = Field(default=None, description="Minimum slider value.")
    label: DynamicString | None = Field(default=None, description="Slider label.")
    checks: list[CheckRule] | None = Field(
        default=None, description="Validation rules."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(extra="forbid")


class DateTimeInputV09(BaseModel):
    """Date and/or time picker."""

    id: str = Field(description="Unique component identifier.")
    component: Literal["DateTimeInput"] = "DateTimeInput"
    value: DynamicString = Field(description="ISO 8601 date/time value.")
    enable_date: bool | None = Field(
        default=None, alias="enableDate", description="Enable date selection."
    )
    enable_time: bool | None = Field(
        default=None, alias="enableTime", description="Enable time selection."
    )
    min: DynamicString | None = Field(
        default=None, description="Minimum allowed date/time."
    )
    max: DynamicString | None = Field(
        default=None, description="Maximum allowed date/time."
    )
    label: DynamicString | None = Field(default=None, description="Input label.")
    checks: list[CheckRule] | None = Field(
        default=None, description="Validation rules."
    )
    weight: float | None = Field(default=None, description="Flex weight.")
    accessibility: AccessibilityAttributes | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class Theme(BaseModel):
    """Surface theme configuration for v0.9.

    Replaces v0.8 ``Styles``.  Adds ``iconUrl`` and ``agentDisplayName``
    for agent attribution; drops ``font``.
    """

    primary_color: str | None = Field(
        default=None,
        alias="primaryColor",
        pattern=r"^#[0-9a-fA-F]{6}$",
        description="Primary brand color as a hex string.",
    )
    icon_url: str | None = Field(
        default=None,
        alias="iconUrl",
        description="URL for an image identifying the agent or tool.",
    )
    agent_display_name: str | None = Field(
        default=None,
        alias="agentDisplayName",
        description="Text label identifying the agent or tool.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class CreateSurface(BaseModel):
    """Signals the client to create a new surface and begin rendering.

    Replaces v0.8 ``BeginRendering``.  ``catalogId`` is now required and
    ``theme`` / ``sendDataModel`` are new.
    """

    surface_id: str = Field(alias="surfaceId", description="Unique surface identifier.")
    catalog_id: str = Field(
        alias="catalogId", description="Catalog identifier for this surface."
    )
    theme: Theme | None = Field(default=None, description="Theme parameters.")
    send_data_model: bool | None = Field(
        default=None,
        alias="sendDataModel",
        description="If true, client sends data model in action metadata.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class UpdateComponents(BaseModel):
    """Updates a surface with a new set of components.

    Replaces v0.8 ``SurfaceUpdate``.  Components use a flat structure where
    ``component`` is a type-name string and properties sit at the top level.
    """

    surface_id: str = Field(alias="surfaceId", description="Target surface identifier.")
    components: list[dict[str, Any]] = Field(
        min_length=1, description="Components to render on the surface."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class UpdateDataModel(BaseModel):
    """Updates the data model for a surface.

    Replaces v0.8 ``DataModelUpdate``.  The ``contents`` adjacency list is
    replaced by a single ``value`` of any JSON type.  ``path`` uses JSON
    Pointer syntax — e.g. ``/user/name``.
    """

    surface_id: str = Field(alias="surfaceId", description="Target surface identifier.")
    path: str | None = Field(
        default=None, description="JSON Pointer path for the update."
    )
    value: Any = Field(
        default=None, description="Value to set. Omit to delete the key."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class DeleteSurfaceV09(BaseModel):
    """Signals the client to delete a surface."""

    surface_id: str = Field(alias="surfaceId", description="Surface to delete.")

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class A2UIMessageV09(BaseModel):
    """Union wrapper for v0.9 server-to-client message types.

    Exactly one message field must be set alongside the ``version`` field.
    """

    version: Literal["v0.9"] = "v0.9"
    create_surface: CreateSurface | None = Field(
        default=None, alias="createSurface", description="Create a new surface."
    )
    update_components: UpdateComponents | None = Field(
        default=None,
        alias="updateComponents",
        description="Update components on a surface.",
    )
    update_data_model: UpdateDataModel | None = Field(
        default=None,
        alias="updateDataModel",
        description="Update the surface data model.",
    )
    delete_surface: DeleteSurfaceV09 | None = Field(
        default=None, alias="deleteSurface", description="Delete a surface."
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    @model_validator(mode="after")
    def _check_exactly_one(self) -> A2UIMessageV09:
        """Enforce the spec's exactly-one-of constraint."""
        fields = [
            self.create_surface,
            self.update_components,
            self.update_data_model,
            self.delete_surface,
        ]
        count = sum(f is not None for f in fields)
        if count != 1:
            raise ValueError(
                f"Exactly one A2UI v0.9 message type must be set, got {count}"
            )
        return self


class ActionEvent(BaseModel):
    """User-initiated action from a component.

    Replaces v0.8 ``UserAction``.  The event field is renamed from
    ``userAction`` to ``action``.
    """

    name: str = Field(description="Action name.")
    surface_id: str = Field(alias="surfaceId", description="Source surface identifier.")
    source_component_id: str = Field(
        alias="sourceComponentId",
        description="Component that triggered the action.",
    )
    timestamp: str = Field(description="ISO 8601 timestamp of the action.")
    context: dict[str, Any] = Field(description="Resolved action context payload.")

    model_config = ConfigDict(populate_by_name=True)


class ClientErrorV09(BaseModel):
    """Structured client-side error report.

    Replaces v0.8's flexible ``ClientError`` with required ``code``,
    ``surfaceId``, and ``message`` fields.
    """

    code: str = Field(description="Error code (e.g. VALIDATION_FAILED).")
    surface_id: str = Field(
        alias="surfaceId", description="Surface where the error occurred."
    )
    message: str = Field(description="Human-readable error description.")
    path: str | None = Field(
        default=None, description="JSON Pointer to the failing field."
    )

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class A2UIEventV09(BaseModel):
    """Union wrapper for v0.9 client-to-server events."""

    version: Literal["v0.9"] = "v0.9"
    action: ActionEvent | None = Field(
        default=None, description="User-initiated action event."
    )
    error: ClientErrorV09 | None = Field(
        default=None, description="Client-side error report."
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _check_exactly_one(self) -> A2UIEventV09:
        """Enforce the spec's exactly-one-of constraint."""
        fields = [self.action, self.error]
        count = sum(f is not None for f in fields)
        if count != 1:
            raise ValueError(
                f"Exactly one A2UI v0.9 event type must be set, got {count}"
            )
        return self


class ClientDataModel(BaseModel):
    """Client data model payload for A2A message metadata.

    When ``sendDataModel`` is ``true`` on ``createSurface``, the client
    attaches this object to every outbound A2A message as
    ``a2uiClientDataModel`` in the metadata.
    """

    version: Literal["v0.9"] = "v0.9"
    surfaces: dict[str, dict[str, Any]] = Field(
        description="Map of surface IDs to their current data models."
    )

    model_config = ConfigDict(extra="forbid")


_V09_KEYS = {"createSurface", "updateComponents", "updateDataModel", "deleteSurface"}


def extract_a2ui_v09_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract JSON objects containing A2UI v0.9 keys from text.

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
            if isinstance(obj, dict) and _V09_KEYS & obj.keys():
                results.append(obj)
            idx = end_idx
        except json.JSONDecodeError:
            idx += 1
    return results

"""A2UI client extension for the A2A protocol."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypeIs, TypedDict

from crewai.a2a.extensions.a2ui.models import extract_a2ui_json_objects
from crewai.a2a.extensions.a2ui.prompt import (
    build_a2ui_system_prompt,
    build_a2ui_v09_system_prompt,
)
from crewai.a2a.extensions.a2ui.server_extension import (
    A2UI_MIME_TYPE,
    A2UI_STANDARD_CATALOG_ID,
    A2UI_V09_BASIC_CATALOG_ID,
)
from crewai.a2a.extensions.a2ui.v0_9 import extract_a2ui_v09_json_objects
from crewai.a2a.extensions.a2ui.validator import (
    A2UIValidationError,
    validate_a2ui_message,
    validate_a2ui_message_v09,
)


if TYPE_CHECKING:
    from a2a.types import Message

    from crewai.agent.core import Agent


logger = logging.getLogger(__name__)


class StylesDict(TypedDict, total=False):
    """Serialized surface styling."""

    font: str
    primaryColor: str


class ComponentEntryDict(TypedDict, total=False):
    """Serialized component entry in a surface update."""

    id: str
    weight: float
    component: dict[str, Any]


class BeginRenderingDict(TypedDict, total=False):
    """Serialized beginRendering payload."""

    surfaceId: str
    root: str
    catalogId: str
    styles: StylesDict


class SurfaceUpdateDict(TypedDict, total=False):
    """Serialized surfaceUpdate payload."""

    surfaceId: str
    components: list[ComponentEntryDict]


class DataEntryDict(TypedDict, total=False):
    """Serialized data model entry."""

    key: str
    valueString: str
    valueNumber: float
    valueBoolean: bool
    valueMap: list[DataEntryDict]


class DataModelUpdateDict(TypedDict, total=False):
    """Serialized dataModelUpdate payload."""

    surfaceId: str
    path: str
    contents: list[DataEntryDict]


class DeleteSurfaceDict(TypedDict):
    """Serialized deleteSurface payload."""

    surfaceId: str


class A2UIMessageDict(TypedDict, total=False):
    """Serialized A2UI v0.8 server-to-client message with exactly one key set."""

    beginRendering: BeginRenderingDict
    surfaceUpdate: SurfaceUpdateDict
    dataModelUpdate: DataModelUpdateDict
    deleteSurface: DeleteSurfaceDict


class ThemeDict(TypedDict, total=False):
    """Serialized v0.9 theme."""

    primaryColor: str
    iconUrl: str
    agentDisplayName: str


class CreateSurfaceDict(TypedDict, total=False):
    """Serialized createSurface payload."""

    surfaceId: str
    catalogId: str
    theme: ThemeDict
    sendDataModel: bool


class UpdateComponentsDict(TypedDict, total=False):
    """Serialized updateComponents payload."""

    surfaceId: str
    components: list[dict[str, Any]]


class UpdateDataModelDict(TypedDict, total=False):
    """Serialized updateDataModel payload."""

    surfaceId: str
    path: str
    value: Any


class DeleteSurfaceV09Dict(TypedDict):
    """Serialized v0.9 deleteSurface payload."""

    surfaceId: str


class A2UIMessageV09Dict(TypedDict, total=False):
    """Serialized A2UI v0.9 server-to-client message with version and exactly one key set."""

    version: Literal["v0.9"]
    createSurface: CreateSurfaceDict
    updateComponents: UpdateComponentsDict
    updateDataModel: UpdateDataModelDict
    deleteSurface: DeleteSurfaceV09Dict


A2UIAnyMessageDict = A2UIMessageDict | A2UIMessageV09Dict


def is_v09_message(msg: A2UIAnyMessageDict) -> TypeIs[A2UIMessageV09Dict]:
    """Narrow a message dict to the v0.9 variant."""
    return msg.get("version") == "v0.9"


def is_v08_message(msg: A2UIAnyMessageDict) -> TypeIs[A2UIMessageDict]:
    """Narrow a message dict to the v0.8 variant."""
    return "version" not in msg


@dataclass
class A2UIConversationState:
    """Tracks active A2UI surfaces and data models across a conversation."""

    active_surfaces: dict[str, dict[str, Any]] = Field(default_factory=dict)
    data_models: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    last_a2ui_messages: list[A2UIAnyMessageDict] = Field(default_factory=list)
    initialized_surfaces: set[str] = Field(default_factory=set)

    def is_ready(self) -> bool:
        """Return True when at least one surface has been initialized via beginRendering."""
        return bool(self.initialized_surfaces)


class A2UIClientExtension:
    """A2A client extension that adds A2UI support to agents.

    Implements the ``A2AExtension`` protocol to inject A2UI prompt
    instructions, track UI state across conversations, and validate
    A2UI messages in responses.

    Example::

        A2AClientConfig(
            endpoint="...",
            extensions=["https://a2ui.org/a2a-extension/a2ui/v0.8"],
            client_extensions=[A2UIClientExtension()],
        )
    """

    def __init__(
        self,
        catalog_id: str | None = None,
        allowed_components: list[str] | None = None,
        version: str = "v0.8",
    ) -> None:
        """Initialize the A2UI client extension.

        Args:
            catalog_id: Catalog identifier to use for prompt generation.
            allowed_components: Subset of component names to expose to the agent.
            version: Protocol version, ``"v0.8"`` or ``"v0.9"``.
        """
        self._catalog_id = catalog_id
        self._allowed_components = allowed_components
        self._version = version

    def inject_tools(self, agent: Agent) -> None:
        """No-op — A2UI uses prompt augmentation rather than tool injection."""

    def extract_state_from_history(
        self, conversation_history: Sequence[Message]
    ) -> A2UIConversationState | None:
        """Scan conversation history for A2UI DataParts and track surface state.

        When ``catalog_id`` is set, only surfaces matching that catalog are tracked.
        """
        state = A2UIConversationState()

        for message in conversation_history:
            for part in message.parts:
                root = part.root
                if root.kind != "data":
                    continue
                metadata = root.metadata or {}
                mime_type = metadata.get("mimeType", "")
                if mime_type != A2UI_MIME_TYPE:
                    continue

                data = root.data
                if not isinstance(data, dict):
                    continue

                surface_id = _get_surface_id(data)
                if not surface_id:
                    continue

                if self._catalog_id and "beginRendering" in data:
                    catalog_id = data["beginRendering"].get("catalogId")
                    if catalog_id and catalog_id != self._catalog_id:
                        continue
                if self._catalog_id and "createSurface" in data:
                    catalog_id = data["createSurface"].get("catalogId")
                    if catalog_id and catalog_id != self._catalog_id:
                        continue

                if "deleteSurface" in data:
                    state.active_surfaces.pop(surface_id, None)
                    state.data_models.pop(surface_id, None)
                    state.initialized_surfaces.discard(surface_id)
                elif "beginRendering" in data:
                    state.initialized_surfaces.add(surface_id)
                    state.active_surfaces[surface_id] = data["beginRendering"]
                elif "createSurface" in data:
                    state.initialized_surfaces.add(surface_id)
                    state.active_surfaces[surface_id] = data["createSurface"]
                elif "surfaceUpdate" in data:
                    if surface_id not in state.initialized_surfaces:
                        logger.warning(
                            "surfaceUpdate for uninitialized surface %s",
                            surface_id,
                        )
                    state.active_surfaces[surface_id] = data["surfaceUpdate"]
                elif "updateComponents" in data:
                    if surface_id not in state.initialized_surfaces:
                        logger.warning(
                            "updateComponents for uninitialized surface %s",
                            surface_id,
                        )
                    state.active_surfaces[surface_id] = data["updateComponents"]
                elif "dataModelUpdate" in data:
                    contents = data["dataModelUpdate"].get("contents", [])
                    state.data_models.setdefault(surface_id, []).extend(contents)
                elif "updateDataModel" in data:
                    update = data["updateDataModel"]
                    state.data_models.setdefault(surface_id, []).append(update)

        if not state.active_surfaces and not state.data_models:
            return None
        return state

    def augment_prompt(
        self,
        base_prompt: str,
        _conversation_state: A2UIConversationState | None,
    ) -> str:
        """Append A2UI system prompt instructions to the base prompt."""
        if self._version == "v0.9":
            a2ui_prompt = build_a2ui_v09_system_prompt(
                catalog_id=self._catalog_id,
                allowed_components=self._allowed_components,
            )
        else:
            a2ui_prompt = build_a2ui_system_prompt(
                catalog_id=self._catalog_id,
                allowed_components=self._allowed_components,
            )
        return f"{base_prompt}\n\n{a2ui_prompt}"

    def process_response(
        self,
        agent_response: Any,
        conversation_state: A2UIConversationState | None,
    ) -> Any:
        """Extract and validate A2UI JSON from agent output.

        When ``allowed_components`` is set, components not in the allowlist are
        logged and stripped from surface updates.  Stores extracted A2UI messages
        on the conversation state and returns the original response unchanged.
        """
        text = (
            agent_response if isinstance(agent_response, str) else str(agent_response)
        )
        results: list[A2UIAnyMessageDict]
        if self._version == "v0.9":
            results = list(_extract_and_validate_v09(text))
            if self._allowed_components:
                allowed = set(self._allowed_components)
                results = [
                    _filter_components_v09(m, allowed)
                    for m in results
                    if is_v09_message(m)
                ]
        else:
            results = list(_extract_and_validate(text))
            if self._allowed_components:
                allowed = set(self._allowed_components)
                results = [
                    _filter_components(msg, allowed)
                    for msg in results
                    if is_v08_message(msg)
                ]

        if results and conversation_state is not None:
            conversation_state.last_a2ui_messages = results

        return agent_response

    def prepare_message_metadata(
        self,
        _conversation_state: A2UIConversationState | None,
    ) -> dict[str, Any]:
        """Inject a2uiClientCapabilities into outbound A2A message metadata.

        Per the A2UI extension spec, clients must declare supported catalog
        IDs in every outbound message's metadata.  v0.9 nests capabilities
        under a ``"v0.9"`` key per ``client_capabilities.json``.
        """
        if self._version == "v0.9":
            default_catalog = A2UI_V09_BASIC_CATALOG_ID
            catalog_ids = [default_catalog]
            if self._catalog_id and self._catalog_id != default_catalog:
                catalog_ids.append(self._catalog_id)
            return {
                "a2uiClientCapabilities": {
                    "v0.9": {
                        "supportedCatalogIds": catalog_ids,
                    },
                },
            }
        catalog_ids = [A2UI_STANDARD_CATALOG_ID]
        if self._catalog_id and self._catalog_id != A2UI_STANDARD_CATALOG_ID:
            catalog_ids.append(self._catalog_id)
        return {
            "a2uiClientCapabilities": {
                "supportedCatalogIds": catalog_ids,
            },
        }


_ALL_SURFACE_ID_KEYS = (
    "beginRendering",
    "surfaceUpdate",
    "dataModelUpdate",
    "deleteSurface",
    "createSurface",
    "updateComponents",
    "updateDataModel",
)


def _get_surface_id(data: dict[str, Any]) -> str | None:
    """Extract surfaceId from any A2UI v0.8 or v0.9 message type."""
    for key in _ALL_SURFACE_ID_KEYS:
        inner = data.get(key)
        if isinstance(inner, dict):
            sid = inner.get("surfaceId")
            if isinstance(sid, str):
                return sid
    return None


def _filter_components(msg: A2UIMessageDict, allowed: set[str]) -> A2UIMessageDict:
    """Strip components whose type is not in *allowed* from a surfaceUpdate."""
    surface_update = msg.get("surfaceUpdate")
    if not isinstance(surface_update, dict):
        return msg

    components = surface_update.get("components")
    if not isinstance(components, list):
        return msg

    filtered = []
    for entry in components:
        component = entry.get("component", {})
        component_types = set(component.keys())
        disallowed = component_types - allowed
        if disallowed:
            logger.debug(
                "Stripping disallowed component type(s) %s from surface update",
                disallowed,
            )
            continue
        filtered.append(entry)

    if len(filtered) == len(components):
        return msg

    return {**msg, "surfaceUpdate": {**surface_update, "components": filtered}}


def _filter_components_v09(
    msg: A2UIMessageV09Dict, allowed: set[str]
) -> A2UIMessageV09Dict:
    """Strip v0.9 components whose type is not in *allowed* from updateComponents.

    v0.9 components use a flat structure where ``component`` is a type-name string.
    """
    update = msg.get("updateComponents")
    if not isinstance(update, dict):
        return msg

    components = update.get("components")
    if not isinstance(components, list):
        return msg

    filtered = []
    for entry in components:
        comp_type = entry.get("component") if isinstance(entry, dict) else None
        if isinstance(comp_type, str) and comp_type not in allowed:
            logger.debug("Stripping disallowed v0.9 component type %s", comp_type)
            continue
        filtered.append(entry)

    if len(filtered) == len(components):
        return msg

    return {**msg, "updateComponents": {**update, "components": filtered}}


def _extract_and_validate(text: str) -> list[A2UIMessageDict]:
    """Extract A2UI v0.8 JSON objects from text and validate them."""
    return [
        dumped
        for candidate in extract_a2ui_json_objects(text)
        if (dumped := _try_validate(candidate)) is not None
    ]


def _try_validate(candidate: dict[str, Any]) -> A2UIMessageDict | None:
    """Validate a single v0.8 A2UI candidate, returning None on failure."""
    try:
        msg = validate_a2ui_message(candidate)
    except A2UIValidationError:
        logger.debug(
            "Skipping invalid A2UI candidate in agent output",
            exc_info=True,
        )
        return None
    return cast(A2UIMessageDict, msg.model_dump(by_alias=True, exclude_none=True))


def _extract_and_validate_v09(text: str) -> list[A2UIMessageV09Dict]:
    """Extract and validate v0.9 A2UI JSON objects from text."""
    return [
        dumped
        for candidate in extract_a2ui_v09_json_objects(text)
        if (dumped := _try_validate_v09(candidate)) is not None
    ]


def _try_validate_v09(candidate: dict[str, Any]) -> A2UIMessageV09Dict | None:
    """Validate a single v0.9 A2UI candidate, returning None on failure."""
    try:
        msg = validate_a2ui_message_v09(candidate)
    except A2UIValidationError:
        logger.debug(
            "Skipping invalid A2UI v0.9 candidate in agent output",
            exc_info=True,
        )
        return None
    return cast(A2UIMessageV09Dict, msg.model_dump(by_alias=True, exclude_none=True))

"""A2UI client extension for the A2A protocol."""

from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING, Any, cast

from pydantic import Field
from pydantic.dataclasses import dataclass
from typing_extensions import TypedDict

from crewai.a2a.extensions.a2ui.models import extract_a2ui_json_objects
from crewai.a2a.extensions.a2ui.prompt import build_a2ui_system_prompt
from crewai.a2a.extensions.a2ui.server_extension import A2UI_MIME_TYPE
from crewai.a2a.extensions.a2ui.validator import (
    A2UIValidationError,
    validate_a2ui_message,
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
    """Serialized A2UI server-to-client message with exactly one key set."""

    beginRendering: BeginRenderingDict
    surfaceUpdate: SurfaceUpdateDict
    dataModelUpdate: DataModelUpdateDict
    deleteSurface: DeleteSurfaceDict


@dataclass
class A2UIConversationState:
    """Tracks active A2UI surfaces and data models across a conversation."""

    active_surfaces: dict[str, dict[str, Any]] = Field(default_factory=dict)
    data_models: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    last_a2ui_messages: list[A2UIMessageDict] = Field(default_factory=list)

    def is_ready(self) -> bool:
        """Return True when at least one surface is active."""
        return bool(self.active_surfaces)


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
    ) -> None:
        """Initialize the A2UI client extension.

        Args:
            catalog_id: Catalog identifier to use for prompt generation.
            allowed_components: Subset of component names to expose to the agent.
        """
        self._catalog_id = catalog_id
        self._allowed_components = allowed_components

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

                if "deleteSurface" in data:
                    state.active_surfaces.pop(surface_id, None)
                    state.data_models.pop(surface_id, None)
                elif "beginRendering" in data:
                    state.active_surfaces[surface_id] = data["beginRendering"]
                elif "surfaceUpdate" in data:
                    state.active_surfaces[surface_id] = data["surfaceUpdate"]
                elif "dataModelUpdate" in data:
                    contents = data["dataModelUpdate"].get("contents", [])
                    state.data_models.setdefault(surface_id, []).extend(contents)

        if not state.active_surfaces and not state.data_models:
            return None
        return state

    def augment_prompt(
        self,
        base_prompt: str,
        _conversation_state: A2UIConversationState | None,
    ) -> str:
        """Append A2UI system prompt instructions to the base prompt."""
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
        a2ui_messages = _extract_and_validate(text)

        if self._allowed_components:
            allowed = set(self._allowed_components)
            a2ui_messages = [_filter_components(msg, allowed) for msg in a2ui_messages]

        if a2ui_messages and conversation_state is not None:
            conversation_state.last_a2ui_messages = a2ui_messages

        return agent_response


def _get_surface_id(data: dict[str, Any]) -> str | None:
    """Extract surfaceId from any A2UI message type."""
    for key in ("beginRendering", "surfaceUpdate", "dataModelUpdate", "deleteSurface"):
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


def _extract_and_validate(text: str) -> list[A2UIMessageDict]:
    """Extract A2UI JSON objects from text and validate them."""
    return [
        dumped
        for candidate in extract_a2ui_json_objects(text)
        if (dumped := _try_validate(candidate)) is not None
    ]


def _try_validate(candidate: dict[str, Any]) -> A2UIMessageDict | None:
    """Validate a single A2UI candidate, returning None on failure."""
    try:
        msg = validate_a2ui_message(candidate)
    except A2UIValidationError:
        logger.debug(
            "Skipping invalid A2UI candidate in agent output",
            exc_info=True,
        )
        return None
    return cast(A2UIMessageDict, msg.model_dump(by_alias=True, exclude_none=True))

"""A2UI client extension for the A2A protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

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


@dataclass
class A2UIConversationState:
    """Tracks active A2UI surfaces and data models across a conversation."""

    active_surfaces: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_models: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    last_a2ui_messages: list[dict[str, Any]] = field(default_factory=list)

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
        """Scan conversation history for A2UI DataParts and track surface state."""
        state = A2UIConversationState()

        for message in conversation_history:
            if not _has_parts(message):
                continue
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
        conversation_state: A2UIConversationState | None,
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

        Stores extracted A2UI messages on the conversation state and returns
        the original response unchanged to preserve the AgentResponseProtocol
        contract.
        """
        text = (
            agent_response if isinstance(agent_response, str) else str(agent_response)
        )
        a2ui_messages = _extract_and_validate(text)

        if a2ui_messages and conversation_state is not None:
            conversation_state.last_a2ui_messages = a2ui_messages

        return agent_response


def _has_parts(message: Any) -> bool:
    """Check if a message has a parts attribute."""
    return isinstance(getattr(message, "parts", None), list)


def _get_surface_id(data: dict[str, Any]) -> str | None:
    """Extract surfaceId from any A2UI message type."""
    for key in ("beginRendering", "surfaceUpdate", "dataModelUpdate", "deleteSurface"):
        inner = data.get(key)
        if isinstance(inner, dict):
            sid = inner.get("surfaceId")
            if isinstance(sid, str):
                return sid
    return None


def _extract_and_validate(text: str) -> list[dict[str, Any]]:
    """Extract A2UI JSON objects from text and validate them."""
    return [
        dumped
        for candidate in extract_a2ui_json_objects(text)
        if (dumped := _try_validate(candidate)) is not None
    ]


def _try_validate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a single A2UI candidate, returning None on failure."""
    try:
        msg = validate_a2ui_message(candidate)
    except A2UIValidationError:
        logger.debug(
            "Skipping invalid A2UI candidate in agent output",
            exc_info=True,
        )
        return None
    return msg.model_dump(by_alias=True, exclude_none=True)

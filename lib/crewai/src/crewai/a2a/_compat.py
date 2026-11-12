"""Compatibility shim for a2a-sdk 0.3.x vs 1.x API differences.

a2a-sdk 1.0 removed and reorganised several public symbols. This module
re-exports them under stable internal paths so the rest of crewai.a2a does
not need per-file try/except guards.

Symbols provided
----------------
From a2a.types (moved to a2a.compat.v0_3.types in 1.x):
  DataPart, FilePart, FileWithBytes, FileWithUri,
  PushNotificationAuthenticationInfo, PushNotificationConfig,
  TaskIdParams, TaskQueryParams, TextPart

From a2a.client.errors (renamed in 1.x):
  A2AClientHTTPError  (was renamed to AgentCardResolutionError)

From a2a.utils (moved to a2a.helpers in 1.x, or removed):
  get_data_parts, get_file_parts, new_agent_text_message,
  new_data_artifact, new_text_artifact

From a2a.utils.errors (removed in 1.x):
  ServerError
"""

from __future__ import annotations

import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Types: TextPart, FileWithBytes, FilePart, DataPart, FileWithUri,
#        PushNotificationConfig, PushNotificationAuthenticationInfo,
#        TaskIdParams, TaskQueryParams
# ---------------------------------------------------------------------------
try:
    # a2a-sdk 0.3.x
    from a2a.types import (  # noqa: F401
        DataPart,
        FilePart,
        FileWithBytes,
        FileWithUri,
        PushNotificationAuthenticationInfo,
        PushNotificationConfig,
        TaskIdParams,
        TaskQueryParams,
        TextPart,
    )
except ImportError:
    # a2a-sdk 1.x moved these to the v0.3 compatibility package
    from a2a.compat.v0_3.types import (  # type: ignore[no-redef]
        DataPart,
        FilePart,
        FileWithBytes,
        FileWithUri,
        PushNotificationAuthenticationInfo,
        PushNotificationConfig,
        TaskIdParams,
        TaskQueryParams,
        TextPart,
    )

# ---------------------------------------------------------------------------
# Errors: A2AClientHTTPError
# ---------------------------------------------------------------------------
try:
    # a2a-sdk 0.3.x
    from a2a.client.errors import A2AClientHTTPError  # noqa: F401
except ImportError:
    # a2a-sdk 1.x renamed this to AgentCardResolutionError
    from a2a.client.errors import (  # type: ignore[no-redef]
        AgentCardResolutionError as A2AClientHTTPError,
    )

# ---------------------------------------------------------------------------
# Utility functions: get_data_parts, get_file_parts, new_agent_text_message,
#                    new_data_artifact, new_text_artifact
# ---------------------------------------------------------------------------
try:
    # a2a-sdk 0.3.x
    from a2a.utils import (  # noqa: F401
        get_data_parts,
        get_file_parts,
        new_agent_text_message,
        new_data_artifact,
        new_text_artifact,
    )
except ImportError:
    # a2a-sdk 1.x moved/removed these; provide local equivalents that use the
    # compat types imported above.
    from a2a.types import Artifact, Message, Part, Role  # type: ignore[assignment]

    def get_data_parts(parts: list[Part]) -> list[dict[str, Any]]:  # type: ignore[misc]
        """Extract data dicts from DataPart objects."""
        return [part.root.data for part in parts if isinstance(part.root, DataPart)]

    def get_file_parts(  # type: ignore[misc]
        parts: list[Part],
    ) -> list[FileWithBytes | FileWithUri]:
        """Extract file objects from FilePart objects."""
        return [part.root.file for part in parts if isinstance(part.root, FilePart)]

    def new_agent_text_message(  # type: ignore[misc]
        text: str,
        context_id: str | None = None,
        task_id: str | None = None,
    ) -> Message:
        """Create an agent message with a single TextPart."""
        return Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text=text))],
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            task_id=task_id,
        )

    def new_text_artifact(  # type: ignore[misc]
        name: str,
        text: str,
        description: str | None = None,
    ) -> Artifact:
        """Create an Artifact containing a single TextPart."""
        return Artifact(
            artifact_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=text))],
            name=name,
            description=description,
        )

    def new_data_artifact(  # type: ignore[misc]
        name: str,
        data: dict[str, Any],
        description: str | None = None,
    ) -> Artifact:
        """Create an Artifact containing a single DataPart."""
        return Artifact(
            artifact_id=str(uuid.uuid4()),
            parts=[Part(root=DataPart(data=data))],
            name=name,
            description=description,
        )

# ---------------------------------------------------------------------------
# ServerError (removed in a2a-sdk 1.x)
# ---------------------------------------------------------------------------
try:
    # a2a-sdk 0.3.x
    from a2a.utils.errors import ServerError  # noqa: F401
except ImportError:
    # a2a-sdk 1.x dropped ServerError; define a local stand-in
    class ServerError(Exception):  # type: ignore[no-redef]
        """Wrapper for A2A server-side errors (compat shim for a2a-sdk 1.x)."""

        def __init__(self, error: Any = None) -> None:
            self.error = error

        def __str__(self) -> str:
            if self.error is None:
                return "None"
            msg = getattr(self.error, "message", None)
            return msg if msg is not None else self.error.__class__.__name__

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.error!r})"


__all__ = [
    "A2AClientHTTPError",
    "DataPart",
    "FilePart",
    "FileWithBytes",
    "FileWithUri",
    "PushNotificationAuthenticationInfo",
    "PushNotificationConfig",
    "ServerError",
    "TaskIdParams",
    "TaskQueryParams",
    "TextPart",
    "get_data_parts",
    "get_file_parts",
    "new_agent_text_message",
    "new_data_artifact",
    "new_text_artifact",
]

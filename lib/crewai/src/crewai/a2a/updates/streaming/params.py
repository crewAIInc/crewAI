"""Common parameter extraction for streaming handlers."""

from __future__ import annotations

from a2a.types import TaskArtifactUpdateEvent, TaskStatusUpdateEvent


def process_status_update(
    update: TaskStatusUpdateEvent,
    result_parts: list[str],
) -> bool:
    """Process a status update event and extract text parts.

    Args:
        update: The status update event.
        result_parts: List to append text parts to (modified in place).

    Returns:
        True if this is a final update, False otherwise.
    """
    is_final = update.final
    if update.status and update.status.message and update.status.message.parts:
        result_parts.extend(
            part.root.text
            for part in update.status.message.parts
            if part.root.kind == "text" and part.root.text
        )
    return is_final


def process_artifact_update(
    update: TaskArtifactUpdateEvent,
    result_parts: list[str],
    artifact_positions: dict[str, int],
) -> None:
    """Add the text parts of an artifact update to the accumulated result.

    A chunk sent with ``append=True`` continues the text of the same artifact,
    so it is joined onto that artifact's last part with no separator. Other
    parts are added as separate entries.

    Args:
        update: The artifact update event.
        result_parts: List of text parts to update (modified in place).
        artifact_positions: Index in ``result_parts`` of each artifact's last
            text part, keyed by artifact ID (modified in place).
    """
    artifact = update.artifact
    for part in artifact.parts:
        if part.root.kind != "text":
            continue
        position = artifact_positions.get(artifact.artifact_id)
        if update.append and position is not None:
            result_parts[position] += part.root.text
        else:
            result_parts.append(part.root.text)
            artifact_positions[artifact.artifact_id] = len(result_parts) - 1

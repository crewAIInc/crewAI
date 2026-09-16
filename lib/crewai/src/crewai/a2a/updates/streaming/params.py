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
    """Add the text of an artifact update to the accumulated result.

    Each artifact keeps one entry in ``result_parts``. A chunk sent with
    ``append=True`` continues that entry with no separator, matching how the
    server built the text. An update sent without ``append`` replaces the
    artifact's text, the way ``a2a.utils.append_artifact_to_task`` replaces an
    artifact of the same ID.

    Args:
        update: The artifact update event.
        result_parts: List of text parts to update (modified in place).
        artifact_positions: Index in ``result_parts`` of each artifact's text,
            keyed by artifact ID (modified in place).
    """
    artifact = update.artifact
    texts = [part.root.text for part in artifact.parts if part.root.kind == "text"]
    if not texts:
        return

    position = artifact_positions.get(artifact.artifact_id)
    if position is None:
        result_parts.append(" ".join(texts))
        artifact_positions[artifact.artifact_id] = len(result_parts) - 1
    elif update.append:
        result_parts[position] += "".join(texts)
    else:
        result_parts[position] = " ".join(texts)

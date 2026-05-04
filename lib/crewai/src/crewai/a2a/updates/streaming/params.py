"""Common parameter extraction for streaming handlers."""

from __future__ import annotations

from a2a.types import TaskStatusUpdateEvent


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

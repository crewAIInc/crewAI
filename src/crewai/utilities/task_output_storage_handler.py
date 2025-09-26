"""Task output storage handler for managing task execution results.

This module provides functionality for storing and retrieving task outputs
from persistent storage, supporting replay and audit capabilities.
"""

from typing import Any

from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)
from crewai.task import Task


class TaskOutputStorageHandler:
    """Manages storage and retrieval of task outputs.

    This handler provides an interface to persist and retrieve task execution
    results, supporting features like replay and audit trails.

    Attributes:
        storage: The underlying SQLite storage implementation.
    """

    def __init__(self) -> None:
        """Initialize the task output storage handler."""
        self.storage = KickoffTaskOutputsSQLiteStorage()

    def update(self, task_index: int, log: dict[str, Any]) -> None:
        """Update an existing task output in storage.

        Args:
            task_index: The index of the task to update.
            log: Dictionary containing task execution details.

        Raises:
            ValueError: If no saved outputs exist.
        """
        saved_outputs = self.load()
        if saved_outputs is None:
            raise ValueError("Logs cannot be None")

        if log.get("was_replayed", False):
            replayed = {
                "task_id": str(log["task"].id),
                "expected_output": log["task"].expected_output,
                "output": log["output"],
                "was_replayed": log["was_replayed"],
                "inputs": log["inputs"],
            }
            self.storage.update(
                task_index,
                **replayed,
            )
        else:
            self.storage.add(**log)

    def add(
        self,
        task: Task,
        output: dict[str, Any],
        task_index: int,
        inputs: dict[str, Any] | None = None,
        was_replayed: bool = False,
    ) -> None:
        """Add a new task output to storage.

        Args:
            task: The task that was executed.
            output: The output produced by the task.
            task_index: The position of the task in execution sequence.
            inputs: Optional input parameters for the task.
            was_replayed: Whether this is a replayed execution.
        """
        inputs = inputs or {}
        self.storage.add(task, output, task_index, was_replayed, inputs)

    def reset(self) -> None:
        """Clear all stored task outputs."""
        self.storage.delete_all()

    def load(self) -> list[dict[str, Any]] | None:
        """Load all stored task outputs.

        Returns:
            List of task output dictionaries, or None if no outputs exist.
        """
        return self.storage.load()

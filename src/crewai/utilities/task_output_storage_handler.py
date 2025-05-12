from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)
from crewai.task import Task

"""Handles storage and retrieval of task execution outputs."""

class ExecutionLog(BaseModel):
    """Represents a log entry for task execution."""

    task_id: str
    expected_output: str | None = None
    output: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    task_index: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    was_replayed: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


"""Manages storage and retrieval of task outputs."""

class TaskOutputStorageHandler:
    def __init__(self) -> None:
        self.storage = KickoffTaskOutputsSQLiteStorage()

    def update(self, task_index: int, log: dict[str, Any]) -> None:
        saved_outputs = self.load()
        if saved_outputs is None:
            msg = "Logs cannot be None"
            raise ValueError(msg)

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
        if inputs is None:
            inputs = {}
        self.storage.add(task, output, task_index, was_replayed, inputs)

    def reset(self) -> None:
        self.storage.delete_all()

    def load(self) -> list[dict[str, Any]] | None:
        return self.storage.load()

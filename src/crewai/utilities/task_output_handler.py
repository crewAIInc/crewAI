import json
import os

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional, List
from crewai.utilities.crew_json_encoder import CrewJSONEncoder


class ExecutionLog(BaseModel):
    task_id: str
    expected_output: Optional[str] = None
    output: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    task_index: int
    inputs: Dict[str, Any] = Field(default_factory=dict)
    was_replayed: bool = False

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class TaskOutputJsonHandler:
    def __init__(self, file_name: str) -> None:
        self.file_path = os.path.join(os.getcwd(), file_name)

    def initialize_file(self) -> None:
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            with open(self.file_path, "w") as file:
                json.dump([], file)

    def update(self, task_index: int, log: ExecutionLog):
        logs = self.load()
        if task_index < len(logs):
            logs[task_index] = log
        else:
            logs.append(log)
        self.save(logs)

    def save(self, logs: List[ExecutionLog]):
        with open(self.file_path, "w") as file:
            json.dump(logs, file, indent=2, cls=CrewJSONEncoder)

    def reset(self):
        """Reset the JSON file by creating an empty file."""
        with open(self.file_path, "w") as f:
            json.dump([], f)

    def load(self) -> List[ExecutionLog]:
        try:
            if (
                not os.path.exists(self.file_path)
                or os.path.getsize(self.file_path) == 0
            ):
                return []

            with open(self.file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File {self.file_path} not found. Returning empty list.")
            return []
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from file {self.file_path}. Returning empty list."
            )
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

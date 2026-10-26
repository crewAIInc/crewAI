from datetime import datetime
import json
import os
import tempfile
from typing import Any, TypedDict

from crewai_core.lock_store import lock as store_lock
from typing_extensions import Unpack


class LogEntry(TypedDict, total=False):
    """TypedDict for log entry kwargs with optional fields for flexibility."""

    task_name: str
    task: str
    agent: str
    status: str
    output: str
    input: str
    message: str
    level: str
    crew: str
    flow: str
    tool: str
    error: str
    duration: float
    metadata: dict[str, Any]


class FileHandler:
    """Handler for file operations supporting both JSON and text-based logging.

    Attributes:
        _path: The path to the log file.
    """

    def __init__(self, file_path: bool | str) -> None:
        """Initialize the FileHandler with the specified file path.
        Args:
            file_path: Path to the log file or boolean flag.
        """
        self._initialize_path(file_path)

    def _initialize_path(self, file_path: bool | str) -> None:
        """Initialize the file path based on the input type.

        Args:
            file_path: Path to the log file or boolean flag.

        Raises:
            ValueError: If file_path is neither a string nor a boolean.
        """
        if file_path is True:
            self._path = os.path.join(os.curdir, "logs.txt")

        elif isinstance(file_path, str):
            if file_path.endswith((".json", ".txt")):
                self._path = file_path
            else:
                self._path = file_path + ".txt"

        else:
            raise ValueError("file_path must be a string or boolean.")

    def log(self, **kwargs: Unpack[LogEntry]) -> None:
        """Log data with structured fields.

        Keyword Args:
            task_name: Name of the task.
            task: Description of the task.
            agent: Name of the agent.
            status: Status of the operation.
            output: Output data.
            input: Input data.
            message: Log message.
            level: Log level (e.g., INFO, ERROR).
            crew: Name of the crew.
            flow: Name of the flow.
            tool: Name of the tool used.
            error: Error message if any.
            duration: Duration of the operation in seconds.
            metadata: Additional metadata as a dictionary.

        Raises:
            ValueError: If logging fails.
        """
        try:
            with store_lock(f"file:{os.path.realpath(self._path)}"):
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = {"timestamp": now, **kwargs}

                if self._path.endswith(".json"):
                    try:
                        with open(self._path, encoding="utf-8") as read_file:
                            existing_data = json.load(read_file)
                            existing_data.append(log_entry)
                    except (json.JSONDecodeError, FileNotFoundError):
                        existing_data = [log_entry]

                    serialized = json.dumps(existing_data, indent=4)
                    with open(self._path, "w", encoding="utf-8") as write_file:
                        write_file.write(serialized)
                        write_file.write("\n")

                else:
                    message = (
                        f"{now}: "
                        + ", ".join(
                            [f'{key}="{value}"' for key, value in kwargs.items()]
                        )
                        + "\n"
                    )
                    with open(self._path, "a", encoding="utf-8") as file:
                        file.write(message)

        except Exception as e:
            raise ValueError(f"Failed to log message: {e!s}") from e


class JSONHandler:
    """Handler for saving and loading JSON data.

    Attributes:
        file_path: The path to the JSON file.
    """

    def __init__(self, file_name: str) -> None:
        """Initialize the JSONHandler with the name of the file where data will be stored.

        The file will be saved in the current directory.

        Args:
            file_name: The name of the file for saving and loading data.
        """
        if not file_name.endswith(".json"):
            raise ValueError("Training data filename must end with .json")

        self.file_path = os.path.join(os.getcwd(), file_name)

    def initialize_file(self) -> None:
        """Initialize the file with an empty dictionary and overwrite any existing data."""
        self.save({})

    def save(self, data: Any) -> None:
        """
        Save the data to the specified file as JSON.

        Args:
          data: The data to be saved to the file.
        """
        with store_lock(f"file:{os.path.realpath(self.file_path)}"):
            file_descriptor, temporary_path = tempfile.mkstemp(
                dir=os.path.dirname(self.file_path),
                prefix=f".{os.path.basename(self.file_path)}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                    json.dump(data, file)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary_path, self.file_path)
            except Exception:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
                raise

    def load(self) -> Any:
        """Load the data from the specified JSON file.

        Returns:
            The data loaded from the file.
        """
        if not os.path.exists(self.file_path):
            return {}

        with store_lock(f"file:{os.path.realpath(self.file_path)}"):
            try:
                with open(self.file_path, encoding="utf-8") as file:
                    return json.load(file)
            except FileNotFoundError:
                return {}
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Training data file contains invalid JSON: {self.file_path}"
                ) from error

from datetime import datetime
import json
import logging
import os
from typing import Any, TypedDict

from typing_extensions import Unpack


logger = logging.getLogger(__name__)


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
        if file_path is True:  # File path is boolean True
            self._path = os.path.join(os.curdir, "logs.txt")

        elif isinstance(file_path, str):  # File path is a string
            if file_path.endswith((".json", ".txt")):
                self._path = (
                    file_path  # No modification if the file ends with .json or .txt
                )
            else:
                self._path = (
                    file_path + ".txt"
                )  # Append .txt if the file doesn't end with .json or .txt

        else:
            raise ValueError(
                "file_path must be a string or boolean."
            )  # Handle the case where file_path isn't valid

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
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {"timestamp": now, **kwargs}

            if self._path.endswith(".json"):
                # Append log in JSON format
                try:
                    # Try reading existing content to avoid overwriting
                    with open(self._path, encoding="utf-8") as read_file:
                        existing_data = json.load(read_file)
                        existing_data.append(log_entry)
                except (json.JSONDecodeError, FileNotFoundError):
                    # If no valid JSON or file doesn't exist, start with an empty list
                    existing_data = [log_entry]

                with open(self._path, "w", encoding="utf-8") as write_file:
                    json.dump(existing_data, write_file, indent=4)
                    write_file.write("\n")

            else:
                # Append log in plain text format
                message = (
                    f"{now}: "
                    + ", ".join([f'{key}="{value}"' for key, value in kwargs.items()])
                    + "\n"
                )
                with open(self._path, "a", encoding="utf-8") as file:
                    file.write(message)

        except Exception as e:
            raise ValueError(f"Failed to log message: {e!s}") from e


class PickleHandler:
    """Handler for saving and loading data using JSON serialization.

    Note: Despite the class name (kept for backward compatibility), this handler
    uses JSON serialization instead of pickle to prevent insecure deserialization
    vulnerabilities (CWE-502).

    Attributes:
        file_path: The path to the JSON data file.
    """

    def __init__(self, file_name: str) -> None:
        """Initialize the PickleHandler with the name of the file where data will be stored.

        The file will be saved in the current directory. Files use JSON format
        for safe serialization. Legacy .pkl files are automatically migrated.

        Args:
            file_name: The name of the file for saving and loading data.
        """
        # Strip old .pkl extension if present and use .json
        if file_name.endswith(".pkl"):
            file_name = file_name[:-4]
        if not file_name.endswith(".json"):
            file_name += ".json"

        self.file_path = os.path.join(os.getcwd(), file_name)

        # Derive legacy .pkl path for migration
        self._legacy_pkl_path = self.file_path.rsplit(".json", 1)[0] + ".pkl"

    def _migrate_legacy_pkl(self) -> dict[str, Any] | None:
        """Attempt to migrate data from a legacy .pkl file to JSON format.

        Returns:
            The migrated data if successful, None otherwise.
        """
        if not os.path.exists(self._legacy_pkl_path):
            return None

        try:
            import pickle

            with open(self._legacy_pkl_path, "rb") as f:
                data = pickle.load(f)  # noqa: S301

            # Save as JSON
            self.save(data)

            # Remove the old pkl file after successful migration
            os.remove(self._legacy_pkl_path)
            logger.info(
                f"Migrated legacy pickle file to JSON: {self._legacy_pkl_path} -> {self.file_path}"
            )
            return data  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(f"Failed to migrate legacy pickle file {self._legacy_pkl_path}: {e}")
            return None

    def initialize_file(self) -> None:
        """Initialize the file with an empty dictionary and overwrite any existing data."""
        self.save({})

    def save(self, data: Any) -> None:
        """Save the data to the specified file using JSON.

        Args:
          data: The data to be saved to the file.
        """
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> Any:
        """Load the data from the specified file using JSON.

        Falls back to migrating legacy .pkl files if the JSON file doesn't exist.

        Returns:
            The data loaded from the file.
        """
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            # Try to migrate from legacy pkl file
            migrated = self._migrate_legacy_pkl()
            if migrated is not None:
                return migrated
            return {}  # Return an empty dictionary if no file exists

        with open(self.file_path, "rb") as file:
            try:
                return json.loads(file.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}  # Return an empty dictionary if the file is corrupted
            except Exception:
                raise  # Raise any other exceptions that occur during loading

from datetime import datetime
import json
import os
import pickle
import tempfile
from typing import Any, TypedDict

import portalocker
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
    """Handler for saving and loading data using pickle with thread-safe file operations.

    This implementation uses file locking and atomic writes to ensure data integrity
    in concurrent environments. Reads use shared locks (allowing multiple concurrent
    readers) while writes use exclusive locks and atomic file replacement.

    Attributes:
        file_path: The path to the pickle file.
        lock_timeout: Timeout in seconds for acquiring file locks.
    """

    def __init__(self, file_name: str, lock_timeout: float = 5.0) -> None:
        """Initialize the PickleHandler with the name of the file where data will be stored.

        The file will be saved in the current directory.

        Args:
            file_name: The name of the file for saving and loading data.
            lock_timeout: Timeout in seconds for acquiring file locks. Defaults to 5.0.
        """
        if not file_name.endswith(".pkl"):
            file_name += ".pkl"

        self.file_path = os.path.join(os.getcwd(), file_name)
        self.lock_timeout = lock_timeout

    def initialize_file(self) -> None:
        """Initialize the file with an empty dictionary and overwrite any existing data."""
        self.save({})

    def save(self, data: Any) -> None:
        """Save the data to the specified file using pickle with atomic writes and exclusive locking.

        This method ensures thread-safety by:
        1. Acquiring an exclusive lock on the target file
        2. Writing to a temporary file in the same directory
        3. Atomically replacing the target file with the temporary file

        Args:
            data: The data to be saved to the file.

        Raises:
            portalocker.LockException: If unable to acquire lock within timeout.
            OSError: If file operations fail.
        """
        dirpath = os.path.dirname(self.file_path) or "."
        os.makedirs(dirpath, exist_ok=True)

        # Create lock file path
        lock_file_path = self.file_path + ".lock"
        
        # Acquire an exclusive lock on a separate lock file
        with portalocker.Lock(
            lock_file_path, mode="w", timeout=self.lock_timeout, flags=portalocker.LOCK_EX
        ):
            # Create a temporary file in the same directory to ensure os.replace is atomic
            tmp = tempfile.NamedTemporaryFile(dir=dirpath, delete=False, suffix=".tmp")
            tmp_name = tmp.name
            try:
                tmp.close()
                # Write to temp file
                with open(tmp_name, "wb") as fh:
                    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
                    fh.flush()
                    os.fsync(fh.fileno())  # Ensure data is written to disk
                
                # On Windows, remove target file if it exists before replace
                if os.path.exists(self.file_path):
                    try:
                        os.remove(self.file_path)
                    except OSError:
                        pass  # File might be locked by another process
                
                # Atomically replace target (or move if target was removed)
                os.replace(tmp_name, self.file_path)
            finally:
                # Clean up tmp if it still exists
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except OSError:
                        pass  # Ignore cleanup errors

    def load(self) -> Any:
        """Load the data from the specified file using pickle with shared locking.

        This method ensures thread-safety by acquiring a shared lock, allowing
        multiple concurrent readers while blocking writers.

        Returns:
            The data loaded from the file, or an empty dictionary if the file
            does not exist or is empty.

        Raises:
            portalocker.LockException: If unable to acquire lock within timeout.
            pickle.UnpicklingError: If the file contains corrupted pickle data.
            EOFError: If the file is truncated or empty during reading.
        """
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return {}  # Return an empty dictionary if the file does not exist or is empty

        # Create lock file path
        lock_file_path = self.file_path + ".lock"
        
        # Shared lock for reading (multiple readers allowed)
        with portalocker.Lock(
            lock_file_path, mode="w", timeout=self.lock_timeout, flags=portalocker.LOCK_SH
        ):
            with open(self.file_path, "rb") as file:
                try:
                    return pickle.load(file)  # noqa: S301
                except EOFError:
                    return {}  # Return an empty dictionary if the file is empty or corrupted
                except Exception:
                    raise  # Raise any other exceptions that occur during loading

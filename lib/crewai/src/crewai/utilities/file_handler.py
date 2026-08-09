from datetime import datetime
import hashlib
import hmac
import json
import os
import pickle
import secrets
import stat
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

                    with open(self._path, "w", encoding="utf-8") as write_file:
                        json.dump(existing_data, write_file, indent=4)
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


class PickleHandler:
    """Handler for saving and loading data using pickle with integrity verification.

    A keyed HMAC-SHA256 signature is written alongside the pickle file on save.
    On load, the signature is verified before deserialization to detect tampering.
    Files without a signature are rejected to prevent loading untrusted data.

    Attributes:
        file_path: The path to the pickle file.
    """

    def __init__(self, file_name: str) -> None:
        """Initialize the PickleHandler with the name of the file where data will be stored.

        The file will be saved in the current directory.

        Args:
            file_name: The name of the file for saving and loading data.
        """
        if not file_name.endswith(".pkl"):
            file_name += ".pkl"

        self.file_path = os.path.join(os.getcwd(), file_name)
        self._key = self._load_or_create_key()

    @property
    def _sig_path(self) -> str:
        """Path to the HMAC signature file."""
        return self.file_path + ".sig"

    def _load_or_create_key(self) -> bytes:
        """Load the HMAC key from the user home directory, or create a new one.

        The key is stored in ``~/.crewai/.hmac_key`` with mode 0600 to keep it
        separate from the working directory where pickle files reside. The
        directory and key file are validated for ownership and restrictive
        permissions before use. Key creation uses an exclusive-create flag so
        concurrent processes cannot overwrite each other's key.

        Returns:
            The 32-byte HMAC key.

        Raises:
            OSError: If the key file cannot be created or permissioned.
            PermissionError: If existing key storage has insecure ownership or mode.
        """
        key_dir = os.path.join(os.path.expanduser("~"), ".crewai")
        key_path = os.path.join(key_dir, ".hmac_key")

        if os.path.exists(key_path):
            if self._validate_key_storage(key_dir, key_path):
                try:
                    with open(key_path, "rb") as f:
                        key = f.read()
                        if len(key) == 32:
                            return key
                    raise ValueError(
                        f"HMAC key file {key_path} exists but has invalid length "
                        f"({len(key)} bytes, expected 32). Remove the file to "
                        "regenerate, or restore from a valid backup."
                    )
                except OSError:
                    pass
            # If validation passed but read failed, fall through to create.

        # Validate directory before first-time key creation: os.makedirs with
        # exist_ok=True does not tighten an existing insecure directory.
        if os.path.exists(key_dir):
            self._validate_key_storage(key_dir, key_path)
        else:
            os.makedirs(key_dir, mode=0o700, exist_ok=False)

        key = secrets.token_bytes(32)

        # Atomic no-clobber creation: O_CREAT|O_EXCL prevents two processes
        # from writing different keys simultaneously. If another process won,
        # load its key instead of using our in-memory copy.
        try:
            fd = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            # Another process created the key between our check and create.
            # Validate and load the installed key.
            if self._validate_key_storage(key_dir, key_path):
                with open(key_path, "rb") as f:
                    installed = f.read()
                if len(installed) == 32:
                    return installed
                raise ValueError(
                    f"HMAC key file {key_path} was created concurrently but has "
                    "invalid length. Remove the file to regenerate."
                ) from None
            raise

        try:
            # Write all bytes, handling short writes from the OS.
            offset = 0
            while offset < len(key):
                offset += os.write(fd, key[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

        return key

    @staticmethod
    def _validate_key_storage(key_dir: str, key_path: str) -> bool:
        """Validate that key storage is owned by the current user and has restrictive permissions.

        Args:
            key_dir: Directory containing the key file.
            key_path: Path to the key file. May not exist yet during first creation.

        Returns:
            True if storage is safe to use.

        Raises:
            PermissionError: If ownership or permissions are insecure.
        """
        dir_stat = os.stat(key_dir)

        current_uid = os.getuid()

        if dir_stat.st_uid != current_uid:
            raise PermissionError(
                f"HMAC key directory {key_dir} is not owned by the current user"
            )

        if os.path.islink(key_dir):
            raise PermissionError("HMAC key directory must not be a symlink")

        dir_mode = stat.S_IMODE(dir_stat.st_mode)

        if dir_mode & 0o077:
            raise PermissionError(
                f"HMAC key directory {key_dir} has insecure mode {oct(dir_mode)}; expected 0700"
            )

        # Validate key file only if it exists (it may not during first creation).
        if os.path.exists(key_path):
            file_stat = os.stat(key_path)

            if file_stat.st_uid != current_uid:
                raise PermissionError(
                    f"HMAC key file {key_path} is not owned by the current user"
                )

            if os.path.islink(key_path):
                raise PermissionError("HMAC key file must not be a symlink")

            file_mode = stat.S_IMODE(file_stat.st_mode)

            if file_mode & 0o077:
                raise PermissionError(
                    f"HMAC key file {key_path} has insecure mode {oct(file_mode)}; expected 0600"
                )

        return True

    def initialize_file(self) -> None:
        """Initialize the file with an empty dictionary and overwrite any existing data."""
        self.save({})

    def save(self, data: Any) -> None:
        """Save the data to the specified file using pickle with HMAC signature.

        Args:
            data: The data to be saved to the file.
        """
        with store_lock(f"file:{os.path.realpath(self.file_path)}"):
            with open(self.file_path, "wb") as f:
                pickle.dump(obj=data, file=f)

            with open(self.file_path, "rb") as f:
                payload = f.read()
            signature = hmac.new(self._key, payload, hashlib.sha256).digest()
            with open(self._sig_path, "wb") as f:
                f.write(signature)

    def load(self) -> Any:
        """Load the data from the specified file with HMAC integrity verification.

        The signature file must exist and match the pickle file's contents.
        Files without a signature are rejected to prevent loading untrusted data.

        Returns:
            The data loaded from the file.

        Raises:
            ValueError: If the signature file is missing or verification fails.
        """
        if not os.path.exists(self.file_path):
            return {}

        with store_lock(f"file:{os.path.realpath(self.file_path)}"):
            with open(self.file_path, "rb") as file:
                payload = file.read()

            if not os.path.exists(self._sig_path):
                raise ValueError(
                    f"Integrity check failed for {self.file_path}: "
                    "no signature file found. Re-save the data to generate one."
                )

            try:
                with open(self._sig_path, "rb") as f:
                    stored_sig = f.read()
            except FileNotFoundError:
                raise ValueError(
                    f"Integrity check failed for {self.file_path}: "
                    "signature file disappeared during loading."
                ) from None

            expected_sig = hmac.new(self._key, payload, hashlib.sha256).digest()

            if not hmac.compare_digest(stored_sig, expected_sig):
                raise ValueError(
                    f"Integrity check failed for {self.file_path}: "
                    "signature mismatch - file may have been tampered with"
                )

            import io

            return pickle.load(io.BytesIO(payload))  # noqa: S301

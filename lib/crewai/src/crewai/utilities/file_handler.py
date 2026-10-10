from datetime import datetime
import hashlib
import hmac
import json
import os
import pickle
import secrets
import stat
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
        permissions before use. Key creation stages the key in a temp file
        and publishes it with a no-clobber hard link, so a concurrent process
        never observes a partially written key.

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
            try:
                os.makedirs(key_dir, mode=0o700, exist_ok=False)
            except FileExistsError:
                # Another process created the directory between our check and
                # creation. Validate the winning directory and continue into
                # the no-clobber key creation path below.
                self._validate_key_storage(key_dir, key_path)

        key = secrets.token_bytes(32)

        # Write-then-publish: stage the key in a temp file inside key_dir
        # (0600 from mkstemp), fsync it, then publish with os.link(), which
        # refuses to overwrite an existing file. A concurrent process can
        # therefore never observe a partially written key: the path either
        # does not exist yet or already holds the complete 32 bytes. On a
        # lost race, load the installed key instead of our in-memory copy.
        #
        # os.write() operates on the fd directly, so no CRT newline
        # translation can corrupt the fixed 32-byte length on any platform.
        fd, tmp_key = tempfile.mkstemp(dir=key_dir, prefix=".hmac_key_tmp_")
        try:
            try:
                # Write all bytes, handling short writes from the OS.
                offset = 0
                while offset < len(key):
                    offset += os.write(fd, key[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                os.link(tmp_key, key_path)
            except FileExistsError:
                # Another process published its key first. Validate and load
                # the installed key, which is guaranteed complete.
                if self._validate_key_storage(key_dir, key_path):
                    with open(key_path, "rb") as f:
                        installed = f.read()
                    if len(installed) == 32:
                        return installed
                    raise ValueError(
                        f"HMAC key file {key_path} has invalid length "
                        f"({len(installed)} bytes, expected 32). Remove the file "
                        "to regenerate, or restore from a valid backup."
                    ) from None
                raise
        finally:
            try:
                os.unlink(tmp_key)
            except OSError:
                pass

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
            PermissionError: If the storage is a symlink, or on POSIX if
                ownership or permissions are insecure.
        """
        is_posix = os.name == "posix"

        # Inspect the paths themselves with lstat() so a symlink cannot pass
        # validation by pointing at a well-formed directory or file. Symlinks
        # are rejected on every platform.
        dir_stat = os.lstat(key_dir)
        if stat.S_ISLNK(dir_stat.st_mode):
            raise PermissionError("HMAC key directory must not be a symlink")

        # Ownership and permission checks are POSIX-only: os.getuid() is not
        # available on Windows, where these semantics do not apply. Resolve it
        # dynamically so the module type-checks on every platform (mypy's
        # Windows stubs omit os.getuid); it only runs on POSIX.
        if is_posix:
            getuid = getattr(os, "getuid", None)
            if getuid is None:
                raise PermissionError(
                    "os.getuid is unavailable; cannot verify HMAC key ownership"
                )
            current_uid = getuid()

            if dir_stat.st_uid != current_uid:
                raise PermissionError(
                    f"HMAC key directory {key_dir} is not owned by the current user"
                )

            dir_mode = stat.S_IMODE(dir_stat.st_mode)

            if dir_mode & 0o077:
                # A pre-existing directory (e.g. ~/.crewai created at 0755 by
                # the skills cache or CLI provider/model caches) must not lock
                # existing users out: tighten it to 0700 instead of raising.
                # Ownership was verified above, so the chmod is ours to make.
                # Fail closed only when hardening is impossible.
                try:
                    os.chmod(key_dir, 0o700)
                except OSError as e:
                    raise PermissionError(
                        f"HMAC key directory {key_dir} has insecure mode "
                        f"{oct(dir_mode)} and could not be hardened: {e}"
                    ) from e
                dir_mode = stat.S_IMODE(os.lstat(key_dir).st_mode)
                if dir_mode & 0o077:
                    raise PermissionError(
                        f"HMAC key directory {key_dir} has insecure mode "
                        f"{oct(dir_mode)}; expected 0700"
                    )

        # Validate key file only if it exists (it may not during first creation).
        if os.path.exists(key_path):
            file_stat = os.lstat(key_path)
            if stat.S_ISLNK(file_stat.st_mode):
                raise PermissionError("HMAC key file must not be a symlink")

            if is_posix:
                if file_stat.st_uid != current_uid:
                    raise PermissionError(
                        f"HMAC key file {key_path} is not owned by the current user"
                    )

                file_mode = stat.S_IMODE(file_stat.st_mode)

                if file_mode & 0o077:
                    raise PermissionError(
                        f"HMAC key file {key_path} has insecure mode {oct(file_mode)}; expected 0600"
                    )

        return True

    def initialize_file(self) -> None:
        """Initialize the file with an empty dictionary and overwrite any existing data."""
        self.save({})

    # On-disk record layout for save()/load(): a 4-byte big-endian signature
    # length, the HMAC-SHA256 signature, then the pickle payload, all written
    # in one atomic _atomic_write. A single atomic write means a crash can
    # never leave a new payload paired with an old signature.
    _RECORD_SIG_LEN = 32

    def save(self, data: Any) -> None:
        """Save the data to the specified file using pickle with HMAC signature.

        The payload is serialized exactly once into a buffer; the signature
        covers those same bytes. Payload and signature are stored in one
        self-contained record written atomically (temp file in the
        destination directory + fsync + rename), so a concurrent reader never
        observes a torn write, a crash between two writes cannot orphan a
        payload from its signature, and a symlink planted at the destination
        is replaced rather than followed.

        Args:
            data: The data to be saved to the file.
        """
        payload = pickle.dumps(data)
        signature = hmac.new(self._key, payload, hashlib.sha256).digest()
        record = len(signature).to_bytes(4, "big") + signature + payload
        with store_lock(f"file:{os.path.realpath(self.file_path)}"):
            self._atomic_write(self.file_path, record)
            # Drop any legacy sidecar signature: the record is now
            # self-contained, and a stale .sig must not shadow it.
            try:
                os.unlink(self._sig_path)
            except OSError:
                pass

    @staticmethod
    def _atomic_write(path: str, data: bytes) -> None:
        """Write bytes to path atomically via temp file + fsync + rename.

        The temp file lives in the destination directory (same filesystem, so
        the rename is atomic) and is created 0600 by mkstemp.
        """
        dest_dir = os.path.dirname(os.path.abspath(path)) or os.curdir
        fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".crewai_pkl_tmp_")
        try:
            try:
                offset = 0
                while offset < len(data):
                    offset += os.write(fd, data[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _read_regular_file(path: str) -> bytes:
        """Read a file that must be a regular file, refusing symlinks.

        The open uses O_NOFOLLOW where the platform provides it, so a symlink
        planted at a predictable path cannot redirect the read, and O_NONBLOCK
        so a planted FIFO cannot block the open; the fstat check then rejects
        anything that is not a regular file (FIFO, socket, device).

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the path is a symlink or not a regular file.
        """
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            raise
        except OSError as e:
            raise ValueError(
                f"Refusing to read {path}: not accessible as a regular file ({e})"
            ) from e
        try:
            file_obj = os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise
        with file_obj:
            if not stat.S_ISREG(os.fstat(file_obj.fileno()).st_mode):
                raise ValueError(f"Refusing to read {path}: not a regular file")
            return file_obj.read()

    def load(self) -> Any:
        """Load the data from the specified file with HMAC integrity verification.

        Reads the single self-contained record written by save(); files in
        the legacy layout (payload plus a separate ``.sig`` sidecar, written
        by older versions) remain readable for migration. Files without a
        verifiable signature are rejected to prevent loading untrusted data.

        Returns:
            The data loaded from the file.

        Raises:
            ValueError: If no verifiable signature exists or verification fails.
        """
        if not os.path.exists(self.file_path):
            return {}

        with store_lock(f"file:{os.path.realpath(self.file_path)}"):
            record = self._read_regular_file(self.file_path)

            # Single-record format: 4-byte big-endian signature length, then
            # the signature, then the pickle payload. A real pickle payload
            # never begins with b"\x00\x00\x00\x20" (no pickle opcode is
            # 0x00), so the length prefix cannot collide with a legacy
            # payload file; a forged prefix still fails HMAC verification.
            if len(record) >= 4 + PickleHandler._RECORD_SIG_LEN:
                sig_len = int.from_bytes(record[:4], "big")
                if sig_len == PickleHandler._RECORD_SIG_LEN:
                    stored_sig = record[4 : 4 + PickleHandler._RECORD_SIG_LEN]
                    payload = record[4 + PickleHandler._RECORD_SIG_LEN :]
                    expected_sig = hmac.new(
                        self._key, payload, hashlib.sha256
                    ).digest()
                    if not hmac.compare_digest(stored_sig, expected_sig):
                        raise ValueError(
                            f"Integrity check failed for {self.file_path}: "
                            "signature mismatch - file may have been tampered with"
                        )
                    return pickle.loads(payload)  # noqa: S301

            # Legacy layout: payload file plus a separate .sig sidecar.
            payload = record
            if not os.path.exists(self._sig_path):
                raise ValueError(
                    f"Integrity check failed for {self.file_path}: "
                    "no signature file found. Re-save the data to generate one."
                )

            try:
                stored_sig = self._read_regular_file(self._sig_path)
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

            return pickle.loads(payload)  # noqa: S301

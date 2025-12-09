from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Final, Literal, cast

from cryptography.fernet import Fernet


_FERNET_KEY_LENGTH: Final[Literal[44]] = 44


class TokenManager:
    """Manages encrypted token storage."""

    def __init__(self, file_path: str = "tokens.enc") -> None:
        """Initialize the TokenManager.

        Args:
            file_path: The file path to store encrypted tokens.
        """
        self.file_path = file_path
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key)

    def _get_or_create_key(self) -> bytes:
        """Get or create the encryption key.

        Returns:
            The encryption key as bytes.
        """
        key_filename: str = "secret.key"

        key = self._read_secure_file(key_filename)
        if key is not None and len(key) == _FERNET_KEY_LENGTH:
            return key

        new_key = Fernet.generate_key()
        if self._atomic_create_secure_file(key_filename, new_key):
            return new_key

        key = self._read_secure_file(key_filename)
        if key is not None and len(key) == _FERNET_KEY_LENGTH:
            return key

        raise RuntimeError("Failed to create or read encryption key")

    def save_tokens(self, access_token: str, expires_at: int) -> None:
        """Save the access token and its expiration time.

        Args:
            access_token: The access token to save.
            expires_at: The UNIX timestamp of the expiration time.
        """
        expiration_time = datetime.fromtimestamp(expires_at)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
        }
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        self._atomic_write_secure_file(self.file_path, encrypted_data)

    def get_token(self) -> str | None:
        """Get the access token if it is valid and not expired.

        Returns:
            The access token if valid and not expired, otherwise None.
        """
        encrypted_data = self._read_secure_file(self.file_path)
        if encrypted_data is None:
            return None

        decrypted_data = self.fernet.decrypt(encrypted_data)
        data = json.loads(decrypted_data)

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now():
            return None

        return cast(str | None, data.get("access_token"))

    def clear_tokens(self) -> None:
        """Clear the stored tokens."""
        self._delete_secure_file(self.file_path)

    @staticmethod
    def _get_secure_storage_path() -> Path:
        """Get the secure storage path based on the operating system.

        Returns:
            The secure storage path.
        """
        if sys.platform == "win32":
            base_path = os.environ.get("LOCALAPPDATA")
        elif sys.platform == "darwin":
            base_path = os.path.expanduser("~/Library/Application Support")
        else:
            base_path = os.path.expanduser("~/.local/share")

        app_name = "crewai/credentials"
        storage_path = Path(base_path) / app_name

        storage_path.mkdir(parents=True, exist_ok=True)

        return storage_path

    def _atomic_create_secure_file(self, filename: str, content: bytes) -> bool:
        """Create a file only if it doesn't exist.

        Args:
            filename: The name of the file.
            content: The content to write.

        Returns:
            True if file was created, False if it already exists.
        """
        storage_path = self._get_secure_storage_path()
        file_path = storage_path / filename

        try:
            fd = os.open(file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, content)
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            return False

    def _atomic_write_secure_file(self, filename: str, content: bytes) -> None:
        """Write content to a secure file.

        Args:
            filename: The name of the file.
            content: The content to write.
        """
        storage_path = self._get_secure_storage_path()
        file_path = storage_path / filename

        fd, temp_path = tempfile.mkstemp(dir=storage_path, prefix=f".{filename}.")
        fd_closed = False
        try:
            os.write(fd, content)
            os.close(fd)
            fd_closed = True
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, file_path)
        except Exception:
            if not fd_closed:
                os.close(fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def _read_secure_file(self, filename: str) -> bytes | None:
        """Read the content of a secure file.

        Args:
            filename: The name of the file.

        Returns:
            The content of the file if it exists, otherwise None.
        """
        storage_path = self._get_secure_storage_path()
        file_path = storage_path / filename

        try:
            with open(file_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def _delete_secure_file(self, filename: str) -> None:
        """Delete a secure file.

        Args:
            filename: The name of the file.
        """
        storage_path = self._get_secure_storage_path()
        file_path = storage_path / filename
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass

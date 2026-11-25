"""Encrypted token storage shared by crewai and crewai-cli."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Final, Literal, cast

from cryptography.fernet import Fernet

from crewai_core.lock_store import lock as store_lock


_FERNET_KEY_LENGTH: Final[Literal[44]] = 44


class TokenManager:
    """Manages encrypted token storage on disk under platform-appropriate paths."""

    def __init__(self, file_path: str = "tokens.enc") -> None:
        """Initialize the TokenManager.

        Args:
            file_path: The file path to store encrypted tokens.
        """
        self.file_path = file_path
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key)

    def _get_or_create_key(self) -> bytes:
        """Get or create the encryption key."""
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

    def save_tokens(
        self,
        access_token: str,
        expires_at: int,
        refresh_token: str | None = None,
        token_url: str | None = None,
    ) -> None:
        """Save the access token, its expiration time and an optional refresh token.

        Args:
            access_token: The access token to save.
            expires_at: The UNIX timestamp of the expiration time.
            refresh_token: The refresh token issued with the access token, if any.
            token_url: The token endpoint that issued ``refresh_token``, the only
                URL it is ever sent to. Required with ``refresh_token``.

        Raises:
            ValueError: If ``refresh_token`` is given without ``token_url``.
        """
        expiration_time = datetime.fromtimestamp(expires_at)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
        }
        if refresh_token is not None:
            if token_url is None:
                raise ValueError("token_url is required to save a refresh token")
            data["refresh_token"] = refresh_token
            data["token_url"] = token_url
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        self._atomic_write_secure_file(self.file_path, encrypted_data)

    def get_token(self, min_remaining_seconds: float = 0) -> str | None:
        """Return the access token if it is valid for ``min_remaining_seconds`` more, else None."""
        data = self._load_tokens()
        if data is None:
            return None

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now() + timedelta(seconds=min_remaining_seconds):
            return None

        return data.get("access_token")

    def get_refresh_token(self) -> tuple[str, str] | None:
        """Return the saved refresh token and the token endpoint it may be sent to.

        Logins saved before refresh tokens were stored, or by a provider that did
        not issue one, return None.
        """
        data = self._load_tokens()
        if data is None or "refresh_token" not in data:
            return None
        return data["refresh_token"], data["token_url"]

    def lock(self) -> AbstractContextManager[None]:
        """Hold a cross-process lock on the token file, so only one process
        spends a rotating refresh token at a time."""
        token_file = self._get_secure_storage_path() / self.file_path
        return store_lock(f"file:{os.path.realpath(token_file)}")

    def clear_tokens(self) -> None:
        """Remove the stored token file (no-op if absent)."""
        self._delete_secure_file(self.file_path)

    def _load_tokens(self) -> dict[str, str] | None:
        """Decrypt the token file, or return None if absent."""
        encrypted_data = self._read_secure_file(self.file_path)
        if encrypted_data is None:
            return None
        return cast(dict[str, str], json.loads(self.fernet.decrypt(encrypted_data)))

    @staticmethod
    def _get_secure_storage_path() -> Path:
        """Platform-appropriate per-user credential directory (mode 0o700)."""
        if sys.platform == "win32":
            base_path = os.environ.get("LOCALAPPDATA")
        elif sys.platform == "darwin":
            base_path = os.path.expanduser("~/Library/Application Support")
        else:
            base_path = os.path.expanduser("~/.local/share")

        app_name = "crewai/credentials"
        storage_path = Path(base_path) / app_name

        storage_path.mkdir(parents=True, exist_ok=True)
        # Enforce the documented 0o700 mode: mkdir is subject to umask and does
        # not adjust the mode of a pre-existing directory, so chmod explicitly.
        try:
            storage_path.chmod(0o700)
        except OSError:
            # Best-effort permission hardening only: some platforms/filesystems
            # may reject chmod here, and token operations should still proceed.
            pass

        return storage_path

    def _atomic_create_secure_file(self, filename: str, content: bytes) -> bool:
        """Create a file only if it doesn't already exist."""
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
        """Write content to a secure file via tempfile + os.replace."""
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
        """Read raw bytes from a secure file, or None if absent."""
        storage_path = self._get_secure_storage_path()
        file_path = storage_path / filename

        try:
            with open(file_path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def _delete_secure_file(self, filename: str) -> None:
        """Delete a secure file (no-op if absent)."""
        storage_path = self._get_secure_storage_path()
        file_path = storage_path / filename
        file_path.unlink(missing_ok=True)

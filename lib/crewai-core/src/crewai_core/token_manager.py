"""Encrypted token storage shared by crewai and crewai-cli."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Final, Literal, cast

from cryptography.fernet import Fernet
import portalocker
import portalocker.exceptions


_FERNET_KEY_LENGTH: Final[Literal[44]] = 44

# Long enough for another process's refresh: a 5s token request plus a 5s key fetch.
_LOCK_TIMEOUT_SECONDS: Final[int] = 15

# A saved login lasts 72 hours from its last use and ends 30 days after `crewai login`.
# Only this client enforces it, so a copy of the token file used elsewhere is not bound by it.
_IDLE_LIMIT: Final = timedelta(hours=72)
_LOGIN_LIMIT: Final = timedelta(days=30)
# The last-use time is rewritten at most this often, so a run that reads the login
# many times rewrites the token file about once.
_LAST_USE_RESOLUTION: Final = timedelta(minutes=1)


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
        keep_login_time: bool = False,
    ) -> None:
        """Save the access token, its expiration time and an optional refresh token.

        A saved login lasts 72 hours from its last use and at most 30 days from
        the login that started it; saving counts as a use.

        Args:
            access_token: The access token to save.
            expires_at: The UNIX timestamp of the expiration time.
            refresh_token: The refresh token issued with the access token, if any.
            token_url: The token endpoint that issued ``refresh_token``, the only
                URL it is ever sent to. Required with ``refresh_token``.
            keep_login_time: These tokens renew the saved login, so its 30 days
                still count from when it started. Otherwise this is a new login.

        Raises:
            ValueError: If ``refresh_token`` is given without ``token_url``.
        """
        now = datetime.now(timezone.utc).isoformat()
        saved = self._load_tokens() if keep_login_time else None
        expiration_time = datetime.fromtimestamp(expires_at)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
            "login_at": (saved or {}).get("login_at", now),
            "last_used_at": now,
        }
        if refresh_token is not None:
            if token_url is None:
                raise ValueError("token_url is required to save a refresh token")
            data["refresh_token"] = refresh_token
            data["token_url"] = token_url
        self._write_tokens(data)

    def record_use(self) -> None:
        """Record that the saved login was just used, keeping it for 72 more hours.

        Best effort: when the token file cannot be locked or written, the token in
        hand is still valid and the login lapses 72 hours after its last recorded use.
        """
        data = self._load_tokens()
        if data is None:
            return
        now = datetime.now(timezone.utc)
        last_used_at = data.get("last_used_at")
        if (
            last_used_at is not None
            and now - datetime.fromisoformat(last_used_at) < _LAST_USE_RESOLUTION
        ):
            return
        try:
            with self.lock():
                # Re-read under the lock: another process may have rotated the refresh token.
                data = self._load_tokens()
                if data is None:
                    return
                # A login saved before these times were recorded starts counting now.
                data.setdefault("login_at", now.isoformat())
                data["last_used_at"] = now.isoformat()
                self._write_tokens(data)
        # Another process holding the lock past the timeout, or a read-only
        # credentials directory such as a container mount.
        except (portalocker.exceptions.BaseLockException, OSError):
            return

    def get_token(self, min_remaining_seconds: float = 0) -> str | None:
        """Return the access token if it is valid for ``min_remaining_seconds`` more, else None.

        Also None once the saved login is past its idle or total limit, even
        when the access token itself has not expired.
        """
        data = self._load_tokens()
        if data is None or not _within_limits(data):
            return None

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now() + timedelta(seconds=min_remaining_seconds):
            return None

        return data.get("access_token")

    def get_refresh_token(self) -> tuple[str, str] | None:
        """Return the saved refresh token and the token endpoint it may be sent to.

        Logins saved before refresh tokens were stored, or by a provider that did
        not issue one, return None, as does a login past its idle or total limit.
        """
        data = self._load_tokens()
        if data is None or "refresh_token" not in data or not _within_limits(data):
            return None
        return data["refresh_token"], data["token_url"]

    @property
    def token_path(self) -> str:
        """The resolved path of the token file, which names this login."""
        return os.path.realpath(self._get_secure_storage_path() / self.file_path)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold a cross-process lock on the token file, so only one process
        spends a rotating refresh token at a time.

        The lock file sits next to the token file, in this user's private
        credentials directory, so another local account cannot create or hold it.

        Raises:
            portalocker.exceptions.LockException: When another process holds
                the lock longer than a refresh takes.
        """
        lock_path = self._get_secure_storage_path() / f"{self.file_path}.lock"
        with portalocker.Lock(lock_path, timeout=_LOCK_TIMEOUT_SECONDS):
            yield

    def clear_tokens(self) -> None:
        """Remove the stored token file (no-op if absent)."""
        self._delete_secure_file(self.file_path)

    def _load_tokens(self) -> dict[str, str] | None:
        """Decrypt the token file, or return None if absent."""
        encrypted_data = self._read_secure_file(self.file_path)
        if encrypted_data is None:
            return None
        return cast(dict[str, str], json.loads(self.fernet.decrypt(encrypted_data)))

    def _write_tokens(self, data: dict[str, str]) -> None:
        self._atomic_write_secure_file(
            self.file_path, self.fernet.encrypt(json.dumps(data).encode())
        )

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


def _within_limits(data: dict[str, str]) -> bool:
    """Whether a saved login is inside 72 hours of its last use and 30 days of
    its start. One saved before these times were recorded counts as fresh until
    its first use records them."""
    if "login_at" not in data or "last_used_at" not in data:
        return True
    now = datetime.now(timezone.utc)
    return (
        now - datetime.fromisoformat(data["last_used_at"]) <= _IDLE_LIMIT
        and now - datetime.fromisoformat(data["login_at"]) <= _LOGIN_LIMIT
    )

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from cryptography.fernet import Fernet


class TokenManager:
    def __init__(self, file_path: str = "tokens.enc") -> None:
        """
        Initialize the TokenManager class.

        :param file_path: The file path to store the encrypted tokens. Default is "tokens.enc".
        """
        self.file_path = file_path
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key)

    def _get_or_create_key(self) -> bytes:
        """
        Get or create the encryption key.

        :return: The encryption key.
        """
        key_filename = "secret.key"
        key = self.read_secure_file(key_filename)

        if key is not None:
            return key

        new_key = Fernet.generate_key()
        self.save_secure_file(key_filename, new_key)
        return new_key

    def save_tokens(self, access_token: str, expires_at: int) -> None:
        """
        Save the access token and its expiration time.

        :param access_token: The access token to save.
        :param expires_at: The UNIX timestamp of the expiration time.
        """
        expiration_time = datetime.fromtimestamp(expires_at)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
        }
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        self.save_secure_file(self.file_path, encrypted_data)

    def get_token(self) -> str | None:
        """
        Get the access token if it is valid and not expired.

        :return: The access token if valid and not expired, otherwise None.
        """
        encrypted_data = self.read_secure_file(self.file_path)
        if encrypted_data is None:
            return None

        decrypted_data = self.fernet.decrypt(encrypted_data)  # type: ignore
        data = json.loads(decrypted_data)

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now():
            return None

        return data["access_token"]

    def clear_tokens(self) -> None:
        """
        Clear the tokens.
        """
        self.delete_secure_file(self.file_path)

    def get_secure_storage_path(self) -> Path:
        """
        Get the secure storage path based on the operating system.

        :return: The secure storage path.
        """
        if sys.platform == "win32":
            # Windows: Use %LOCALAPPDATA%
            base_path = os.environ.get("LOCALAPPDATA")
        elif sys.platform == "darwin":
            # macOS: Use ~/Library/Application Support
            base_path = os.path.expanduser("~/Library/Application Support")
        else:
            # Linux and other Unix-like: Use ~/.local/share
            base_path = os.path.expanduser("~/.local/share")

        app_name = "crewai/credentials"
        storage_path = Path(base_path) / app_name

        storage_path.mkdir(parents=True, exist_ok=True)

        return storage_path

    def save_secure_file(self, filename: str, content: bytes) -> None:
        """
        Save the content to a secure file.

        :param filename: The name of the file.
        :param content: The content to save.
        """
        storage_path = self.get_secure_storage_path()
        file_path = storage_path / filename

        with open(file_path, "wb") as f:
            f.write(content)

        # Set appropriate permissions (read/write for owner only)
        os.chmod(file_path, 0o600)

    def read_secure_file(self, filename: str) -> bytes | None:
        """
        Read the content of a secure file.

        :param filename: The name of the file.
        :return: The content of the file if it exists, otherwise None.
        """
        storage_path = self.get_secure_storage_path()
        file_path = storage_path / filename

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return f.read()

    def delete_secure_file(self, filename: str) -> None:
        """
        Delete the secure file.

        :param filename: The name of the file.
        """
        storage_path = self.get_secure_storage_path()
        file_path = storage_path / filename
        if file_path.exists():
            file_path.unlink(missing_ok=True)

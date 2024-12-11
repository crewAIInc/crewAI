import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from auth0.authentication.token_verifier import (
    AsymmetricSignatureVerifier,
    TokenVerifier,
)
from cryptography.fernet import Fernet

from .constants import AUTH0_CLIENT_ID, AUTH0_DOMAIN


def validate_token(id_token: str) -> None:
    """
    Verify the token and its precedence

    :param id_token:
    """
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    issuer = f"https://{AUTH0_DOMAIN}/"
    signature_verifier = AsymmetricSignatureVerifier(jwks_url)
    token_verifier = TokenVerifier(
        signature_verifier=signature_verifier, issuer=issuer, audience=AUTH0_CLIENT_ID
    )
    token_verifier.verify(id_token)


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

    def save_tokens(self, access_token: str, expires_in: int) -> None:
        """
        Save the access token and its expiration time.

        :param access_token: The access token to save.
        :param expires_in: The expiration time of the access token in seconds.
        """
        expiration_time = datetime.now() + timedelta(seconds=expires_in)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
        }
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        self.save_secure_file(self.file_path, encrypted_data)

    def get_token(self) -> Optional[str]:
        """
        Get the access token if it is valid and not expired.

        :return: The access token if valid and not expired, otherwise None.
        """
        encrypted_data = self.read_secure_file(self.file_path)

        decrypted_data = self.fernet.decrypt(encrypted_data)  # type: ignore
        data = json.loads(decrypted_data)

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now():
            return None

        return data["access_token"]

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

    def read_secure_file(self, filename: str) -> Optional[bytes]:
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

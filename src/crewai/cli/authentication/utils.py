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
    def __init__(self, file_path="tokens.enc"):
        self.file_path = file_path
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key)

    def _get_or_create_key(self):
        key_filename = "secret.key"

        if self.read_secure_file(key_filename):
            key = self.read_secure_file(key_filename)
        else:
            key = Fernet.generate_key()
            self.save_secure_file(key_filename, key)
        return key

    def save_tokens(self, access_token, expires_in):
        expiration_time = datetime.now() + timedelta(seconds=expires_in)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
        }
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        self.save_secure_file(self.file_path, encrypted_data)

    def get_token(self) -> Optional[str]:
        encrypted_data = self.read_secure_file(self.file_path)

        decrypted_data = self.fernet.decrypt(encrypted_data)
        data = json.loads(decrypted_data)

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now():
            return None

        return data["access_token"]

    def get_secure_storage_path(self):
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

    def save_secure_file(self, filename, content):
        storage_path = self.get_secure_storage_path()
        file_path = storage_path / filename

        with open(file_path, "wb") as f:
            f.write(content)

        # Set appropriate permissions (read/write for owner only)
        os.chmod(file_path, 0o600)

    def read_secure_file(self, filename):
        storage_path = self.get_secure_storage_path()
        file_path = storage_path / filename

        if not file_path.exists():
            return None

        with open(file_path, "rb") as f:
            return f.read()

import json
import os
from datetime import datetime, timedelta
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
        key_file = "secret.key"
        if os.path.exists(key_file):
            return open(key_file, "rb").read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as key_file:
                key_file.write(key)
            return key

    def save_tokens(self, access_token, expires_in):
        expiration_time = datetime.now() + timedelta(seconds=expires_in)
        data = {
            "access_token": access_token,
            "expiration": expiration_time.isoformat(),
        }
        encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
        with open(self.file_path, "wb") as file:
            file.write(encrypted_data)

    def get_token(self) -> Optional[str]:
        if not os.path.exists(self.file_path):
            return None

        with open(self.file_path, "rb") as file:
            encrypted_data = file.read()

        decrypted_data = self.fernet.decrypt(encrypted_data)
        data = json.loads(decrypted_data)

        expiration = datetime.fromisoformat(data["expiration"])
        if expiration <= datetime.now():
            return None

        return data["access_token"]

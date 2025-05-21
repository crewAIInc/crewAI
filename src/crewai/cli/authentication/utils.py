import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
from cryptography.fernet import Fernet
import jwt
from jwt import PyJWKClient

from .constants import (
    WORKOS_CLIENT_ID,
    WORKOS_DOMAIN,
    WORKOS_ENVIRONMENT_ID,
    WORKOS_TOKEN_URL,
)


def get_auth_token_with_refresh_token(refresh_token: str) -> dict:
    """
    Get an access token using a refresh token.

    :param refresh_token: The refresh token to use.
    :return: A dictionary containing the access token, its expiration time, and a new refresh token, or an empty dictionary if the attempt to get a new access token failed.
    """

    response = requests.post(
        WORKOS_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": WORKOS_CLIENT_ID,
        },
        timeout=10,
    )

    if response.status_code != 200:
        return {}

    data = response.json()
    try:
        validate_token(data.get("access_token"))
    except Exception:
        return {}

    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "expires_in": data.get("expires_in"),
    }


def validate_token(jwt_token: str, token_type: str = "access_token") -> dict:
    """
    Verify the token's signature and claims using PyJWT.

    :param jwt_token: The JWT (JWS) string to validate.
    :return: The decoded token.
    :raises Exception: If the token is invalid for any reason (e.g., signature mismatch,
                       expired, incorrect issuer/audience, JWKS fetching error,
                       missing required claims).
    """

    supported_audiences = {
        "access_token": WORKOS_ENVIRONMENT_ID,
        "id_token": WORKOS_CLIENT_ID,
    }

    jwks_url = f"https://{WORKOS_DOMAIN}/oauth2/jwks"
    expected_issuer = f"https://{WORKOS_DOMAIN}"
    expected_audience = supported_audiences[token_type]
    decoded_token = None

    try:
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(jwt_token)

        decoded_token = jwt.decode(
            jwt_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=expected_audience,
            issuer=expected_issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )
        return decoded_token

    except jwt.ExpiredSignatureError:
        raise Exception("Token has expired.")
    except jwt.InvalidAudienceError:
        raise Exception(f"Invalid token audience. Expected: '{expected_audience}'")
    except jwt.InvalidIssuerError:
        raise Exception(f"Invalid token issuer. Expected: '{expected_issuer}'")
    except jwt.MissingRequiredClaimError as e:
        raise Exception(f"Token is missing required claims: {str(e)}")
    except jwt.exceptions.PyJWKClientError as e:
        raise Exception(f"JWKS or key processing error: {str(e)}")
    except jwt.InvalidTokenError as e:
        raise Exception(f"Invalid token: {str(e)}")


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

    def save_access_token(self, access_token: str, expires_in: int) -> None:
        """
        Save the access token and its expiration time.

        :param access_token: The access token to save.
        :param expires_in: The expiration time of the access token in seconds.
        """
        self._save_token("access_token", access_token, expires_in)

    def save_refresh_token(self, refresh_token: str) -> None:
        """
        Save the refresh token and its expiration time.

        :param refresh_token: The refresh token to save.

        Refresh tokens don't have an expiration time, so the expiration time is set to 100 years from now.
        """
        self._save_token("refresh_token", refresh_token, 3153600000)

    def get_token(self, token_type: str = "access_token") -> Optional[str]:
        """
        Get the specified token if it exists and is valid (not expired).

        :return: The specified token if it exists and hasn't expired, otherwise None.
        """
        encrypted_data = self.read_secure_file(self.file_path)

        decrypted_data = self.fernet.decrypt(encrypted_data)  # type: ignore
        all_tokens = json.loads(decrypted_data)
        if not (token_data := all_tokens.get(token_type)):
            return None

        expiration = datetime.fromisoformat(token_data["expiration"])
        if expiration <= datetime.now():
            return None

        return token_data["value"]

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

    def _save_token(self, token_type: str, token: str, expires_in: int) -> None:
        """
        Save the token and its expiration time, updating the existing token file.
        """
        all_tokens = {}
        raw_existing_data = self.read_secure_file(self.file_path)

        if raw_existing_data:
            try:
                decrypted_data = self.fernet.decrypt(raw_existing_data)
                all_tokens = json.loads(decrypted_data.decode())
            except Exception:
                print("Error decrypting existing token file. Creating new file.")
                all_tokens = {}

        expiration_time = datetime.now() + timedelta(seconds=expires_in)

        all_tokens[token_type] = {
            "value": token,
            "expiration": expiration_time.isoformat(),
        }

        updated_encrypted_data = self.fernet.encrypt(json.dumps(all_tokens).encode())
        self.save_secure_file(self.file_path, updated_encrypted_data)

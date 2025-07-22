import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import jwt
from jwt import PyJWKClient
from cryptography.fernet import Fernet


def validate_jwt_token(
    jwt_token: str, jwks_url: str, issuer: str, audience: str
) -> dict:
    """
    Verify the token's signature and claims using PyJWT.
    :param jwt_token: The JWT (JWS) string to validate.
    :param jwks_url: The URL of the JWKS endpoint.
    :param issuer: The expected issuer of the token.
    :param audience: The expected audience of the token.
    :return: The decoded token.
    :raises Exception: If the token is invalid for any reason (e.g., signature mismatch,
                       expired, incorrect issuer/audience, JWKS fetching error,
                       missing required claims).
    """

    decoded_token = None

    try:
        jwk_client = PyJWKClient(jwks_url)
        signing_key = jwk_client.get_signing_key_from_jwt(jwt_token)

        _unverified_decoded_token = jwt.decode(
            jwt_token, options={"verify_signature": False}
        )
        decoded_token = jwt.decode(
            jwt_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
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
        actual_audience = _unverified_decoded_token.get("aud", "[no audience found]")
        raise Exception(
            f"Invalid token audience. Got: '{actual_audience}'. Expected: '{audience}'"
        )
    except jwt.InvalidIssuerError:
        actual_issuer = _unverified_decoded_token.get("iss", "[no issuer found]")
        raise Exception(
            f"Invalid token issuer. Got: '{actual_issuer}'. Expected: '{issuer}'"
        )
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

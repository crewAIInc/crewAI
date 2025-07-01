import json
import jwt
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from crewai.cli.authentication.utils import TokenManager, validate_jwt_token


@patch("crewai.cli.authentication.utils.PyJWKClient", return_value=MagicMock())
@patch("crewai.cli.authentication.utils.jwt")
class TestValidateToken(unittest.TestCase):
    def test_validate_jwt_token(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.return_value = {"exp": 1719859200}

        # Create signing key object mock with a .key attribute
        mock_pyjwkclient.return_value.get_signing_key_from_jwt.return_value = MagicMock(
            key="mock_signing_key"
        )

        decoded_token = validate_jwt_token(
            jwt_token="aaaaa.bbbbbb.cccccc",
            jwks_url="https://mock_jwks_url",
            issuer="https://mock_issuer",
            audience="app_id_xxxx",
        )

        mock_jwt.decode.assert_called_once_with(
            "aaaaa.bbbbbb.cccccc",
            "mock_signing_key",
            algorithms=["RS256"],
            audience="app_id_xxxx",
            issuer="https://mock_issuer",
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_nbf": True,
                "verify_iat": True,
                "require": ["exp", "iat", "iss", "aud", "sub"],
            },
        )
        mock_pyjwkclient.assert_called_once_with("https://mock_jwks_url")
        self.assertEqual(decoded_token, {"exp": 1719859200})

    def test_validate_jwt_token_expired(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.ExpiredSignatureError
        with self.assertRaises(Exception):
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_invalid_audience(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.InvalidAudienceError
        with self.assertRaises(Exception):
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_invalid_issuer(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.InvalidIssuerError
        with self.assertRaises(Exception):
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_missing_required_claims(
        self, mock_jwt, mock_pyjwkclient
    ):
        mock_jwt.decode.side_effect = jwt.MissingRequiredClaimError
        with self.assertRaises(Exception):
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_jwks_error(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.exceptions.PyJWKClientError
        with self.assertRaises(Exception):
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )

    def test_validate_jwt_token_invalid_token(self, mock_jwt, mock_pyjwkclient):
        mock_jwt.decode.side_effect = jwt.InvalidTokenError
        with self.assertRaises(Exception):
            validate_jwt_token(
                jwt_token="aaaaa.bbbbbb.cccccc",
                jwks_url="https://mock_jwks_url",
                issuer="https://mock_issuer",
                audience="app_id_xxxx",
            )


class TestTokenManager(unittest.TestCase):
    def setUp(self):
        self.token_manager = TokenManager()

    @patch("crewai.cli.authentication.utils.TokenManager.read_secure_file")
    @patch("crewai.cli.authentication.utils.TokenManager.save_secure_file")
    @patch("crewai.cli.authentication.utils.TokenManager._get_or_create_key")
    def test_get_or_create_key_existing(self, mock_get_or_create, mock_save, mock_read):
        mock_key = Fernet.generate_key()
        mock_get_or_create.return_value = mock_key

        token_manager = TokenManager()
        result = token_manager.key

        self.assertEqual(result, mock_key)

    @patch("crewai.cli.authentication.utils.Fernet.generate_key")
    @patch("crewai.cli.authentication.utils.TokenManager.read_secure_file")
    @patch("crewai.cli.authentication.utils.TokenManager.save_secure_file")
    def test_get_or_create_key_new(self, mock_save, mock_read, mock_generate):
        mock_key = b"new_key"
        mock_read.return_value = None
        mock_generate.return_value = mock_key

        result = self.token_manager._get_or_create_key()

        self.assertEqual(result, mock_key)
        mock_read.assert_called_once_with("secret.key")
        mock_generate.assert_called_once()
        mock_save.assert_called_once_with("secret.key", mock_key)

    @patch("crewai.cli.authentication.utils.TokenManager.save_secure_file")
    def test_save_tokens(self, mock_save):
        access_token = "test_token"
        expires_at = int((datetime.now() + timedelta(seconds=3600)).timestamp())

        self.token_manager.save_tokens(access_token, expires_at)

        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        self.assertEqual(args[0], "tokens.enc")
        decrypted_data = self.token_manager.fernet.decrypt(args[1])
        data = json.loads(decrypted_data)
        self.assertEqual(data["access_token"], access_token)
        expiration = datetime.fromisoformat(data["expiration"])
        self.assertEqual(expiration, datetime.fromtimestamp(expires_at))

    @patch("crewai.cli.authentication.utils.TokenManager.read_secure_file")
    def test_get_token_valid(self, mock_read):
        access_token = "test_token"
        expiration = (datetime.now() + timedelta(hours=1)).isoformat()
        data = {"access_token": access_token, "expiration": expiration}
        encrypted_data = self.token_manager.fernet.encrypt(json.dumps(data).encode())
        mock_read.return_value = encrypted_data

        result = self.token_manager.get_token()

        self.assertEqual(result, access_token)

    @patch("crewai.cli.authentication.utils.TokenManager.read_secure_file")
    def test_get_token_expired(self, mock_read):
        access_token = "test_token"
        expiration = (datetime.now() - timedelta(hours=1)).isoformat()
        data = {"access_token": access_token, "expiration": expiration}
        encrypted_data = self.token_manager.fernet.encrypt(json.dumps(data).encode())
        mock_read.return_value = encrypted_data

        result = self.token_manager.get_token()

        self.assertIsNone(result)

    @patch("crewai.cli.authentication.utils.TokenManager.get_secure_storage_path")
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    @patch("crewai.cli.authentication.utils.os.chmod")
    def test_save_secure_file(self, mock_chmod, mock_open, mock_get_path):
        mock_path = MagicMock()
        mock_get_path.return_value = mock_path
        filename = "test_file.txt"
        content = b"test_content"

        self.token_manager.save_secure_file(filename, content)

        mock_path.__truediv__.assert_called_once_with(filename)
        mock_open.assert_called_once_with(mock_path.__truediv__.return_value, "wb")
        mock_open().write.assert_called_once_with(content)
        mock_chmod.assert_called_once_with(mock_path.__truediv__.return_value, 0o600)

    @patch("crewai.cli.authentication.utils.TokenManager.get_secure_storage_path")
    @patch(
        "builtins.open", new_callable=unittest.mock.mock_open, read_data=b"test_content"
    )
    def test_read_secure_file_exists(self, mock_open, mock_get_path):
        mock_path = MagicMock()
        mock_get_path.return_value = mock_path
        mock_path.__truediv__.return_value.exists.return_value = True
        filename = "test_file.txt"

        result = self.token_manager.read_secure_file(filename)

        self.assertEqual(result, b"test_content")
        mock_path.__truediv__.assert_called_once_with(filename)
        mock_open.assert_called_once_with(mock_path.__truediv__.return_value, "rb")

    @patch("crewai.cli.authentication.utils.TokenManager.get_secure_storage_path")
    def test_read_secure_file_not_exists(self, mock_get_path):
        mock_path = MagicMock()
        mock_get_path.return_value = mock_path
        mock_path.__truediv__.return_value.exists.return_value = False
        filename = "test_file.txt"

        result = self.token_manager.read_secure_file(filename)

        self.assertIsNone(result)
        mock_path.__truediv__.assert_called_once_with(filename)

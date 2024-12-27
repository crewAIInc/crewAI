import json
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from crewai.cli.authentication.utils import TokenManager, validate_token
from cryptography.fernet import Fernet


class TestValidateToken(unittest.TestCase):
    @patch("crewai.cli.authentication.utils.AsymmetricSignatureVerifier")
    @patch("crewai.cli.authentication.utils.TokenVerifier")
    def test_validate_token(self, mock_token_verifier, mock_asymmetric_verifier):
        mock_verifier_instance = mock_token_verifier.return_value
        mock_id_token = "mock_id_token"

        validate_token(mock_id_token)

        mock_asymmetric_verifier.assert_called_once_with(
            "https://crewai.us.auth0.com/.well-known/jwks.json"
        )
        mock_token_verifier.assert_called_once_with(
            signature_verifier=mock_asymmetric_verifier.return_value,
            issuer="https://crewai.us.auth0.com/",
            audience="DEVC5Fw6NlRoSzmDCcOhVq85EfLBjKa8",
        )
        mock_verifier_instance.verify.assert_called_once_with(mock_id_token)


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
        expires_in = 3600

        self.token_manager.save_tokens(access_token, expires_in)

        mock_save.assert_called_once()
        args = mock_save.call_args[0]
        self.assertEqual(args[0], "tokens.enc")
        decrypted_data = self.token_manager.fernet.decrypt(args[1])
        data = json.loads(decrypted_data)
        self.assertEqual(data["access_token"], access_token)
        expiration = datetime.fromisoformat(data["expiration"])
        self.assertAlmostEqual(
            expiration,
            datetime.now() + timedelta(seconds=expires_in),
            delta=timedelta(seconds=1),
        )

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

"""Tests for TokenManager with atomic file operations."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet

from crewai.cli.shared.token_manager import TokenManager


class TestTokenManager(unittest.TestCase):
    """Test cases for TokenManager."""

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def setUp(self, mock_get_key: unittest.mock.MagicMock) -> None:
        """Set up test fixtures."""
        mock_get_key.return_value = Fernet.generate_key()
        self.token_manager = TokenManager()

    @patch("crewai.cli.shared.token_manager.TokenManager._read_secure_file")
    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_get_or_create_key_existing(
        self,
        mock_get_or_create: unittest.mock.MagicMock,
        mock_read: unittest.mock.MagicMock,
    ) -> None:
        """Test that existing key is returned when present."""
        mock_key = Fernet.generate_key()
        mock_get_or_create.return_value = mock_key

        token_manager = TokenManager()
        result = token_manager.key

        self.assertEqual(result, mock_key)

    def test_get_or_create_key_new(self) -> None:
        """Test that new key is created when none exists."""
        mock_key = Fernet.generate_key()

        with (
            patch.object(self.token_manager, "_read_secure_file", return_value=None) as mock_read,
            patch.object(self.token_manager, "_atomic_create_secure_file", return_value=True) as mock_atomic_create,
            patch("crewai.cli.shared.token_manager.Fernet.generate_key", return_value=mock_key) as mock_generate,
        ):
            result = self.token_manager._get_or_create_key()

            self.assertEqual(result, mock_key)
            mock_read.assert_called_with("secret.key")
            mock_generate.assert_called_once()
            mock_atomic_create.assert_called_once_with("secret.key", mock_key)

    def test_get_or_create_key_race_condition(self) -> None:
        """Test that another process's key is used when atomic create fails."""
        our_key = Fernet.generate_key()
        their_key = Fernet.generate_key()

        with (
            patch.object(self.token_manager, "_read_secure_file", side_effect=[None, their_key]) as mock_read,
            patch.object(self.token_manager, "_atomic_create_secure_file", return_value=False) as mock_atomic_create,
            patch("crewai.cli.shared.token_manager.Fernet.generate_key", return_value=our_key),
        ):
            result = self.token_manager._get_or_create_key()

            self.assertEqual(result, their_key)
            self.assertEqual(mock_read.call_count, 2)

    @patch("crewai.cli.shared.token_manager.TokenManager._atomic_write_secure_file")
    def test_save_tokens(
        self, mock_write: unittest.mock.MagicMock
    ) -> None:
        """Test saving tokens encrypts and writes atomically."""
        access_token = "test_token"
        expires_at = int((datetime.now() + timedelta(seconds=3600)).timestamp())

        self.token_manager.save_tokens(access_token, expires_at)

        mock_write.assert_called_once()
        args = mock_write.call_args[0]
        self.assertEqual(args[0], "tokens.enc")
        decrypted_data = self.token_manager.fernet.decrypt(args[1])
        data = json.loads(decrypted_data)
        self.assertEqual(data["access_token"], access_token)
        expiration = datetime.fromisoformat(data["expiration"])
        self.assertEqual(expiration, datetime.fromtimestamp(expires_at))

    @patch("crewai.cli.shared.token_manager.TokenManager._read_secure_file")
    def test_get_token_valid(
        self, mock_read: unittest.mock.MagicMock
    ) -> None:
        """Test getting a valid non-expired token."""
        access_token = "test_token"
        expiration = (datetime.now() + timedelta(hours=1)).isoformat()
        data = {"access_token": access_token, "expiration": expiration}
        encrypted_data = self.token_manager.fernet.encrypt(json.dumps(data).encode())
        mock_read.return_value = encrypted_data

        result = self.token_manager.get_token()

        self.assertEqual(result, access_token)

    @patch("crewai.cli.shared.token_manager.TokenManager._read_secure_file")
    def test_get_token_expired(
        self, mock_read: unittest.mock.MagicMock
    ) -> None:
        """Test that expired token returns None."""
        access_token = "test_token"
        expiration = (datetime.now() - timedelta(hours=1)).isoformat()
        data = {"access_token": access_token, "expiration": expiration}
        encrypted_data = self.token_manager.fernet.encrypt(json.dumps(data).encode())
        mock_read.return_value = encrypted_data

        result = self.token_manager.get_token()

        self.assertIsNone(result)

    @patch("crewai.cli.shared.token_manager.TokenManager._read_secure_file")
    def test_get_token_not_found(
        self, mock_read: unittest.mock.MagicMock
    ) -> None:
        """Test that missing token file returns None."""
        mock_read.return_value = None

        result = self.token_manager.get_token()

        self.assertIsNone(result)

    @patch("crewai.cli.shared.token_manager.TokenManager._delete_secure_file")
    def test_clear_tokens(
        self, mock_delete: unittest.mock.MagicMock
    ) -> None:
        """Test clearing tokens deletes the token file."""
        self.token_manager.clear_tokens()

        mock_delete.assert_called_once_with("tokens.enc")


class TestAtomicFileOperations(unittest.TestCase):
    """Test atomic file operations directly."""

    def setUp(self) -> None:
        """Set up test fixtures with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_get_path = TokenManager._get_secure_storage_path

        # Patch to use temp directory
        def mock_get_path() -> Path:
            return Path(self.temp_dir)

        TokenManager._get_secure_storage_path = staticmethod(mock_get_path)

    def tearDown(self) -> None:
        """Clean up temp directory."""
        TokenManager._get_secure_storage_path = staticmethod(self.original_get_path)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_atomic_create_new_file(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test atomic create succeeds for new file."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        result = tm._atomic_create_secure_file("test.txt", b"content")

        self.assertTrue(result)
        file_path = Path(self.temp_dir) / "test.txt"
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_bytes(), b"content")
        self.assertEqual(file_path.stat().st_mode & 0o777, 0o600)

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_atomic_create_existing_file(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test atomic create fails for existing file."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        # Create file first
        file_path = Path(self.temp_dir) / "test.txt"
        file_path.write_bytes(b"original")

        result = tm._atomic_create_secure_file("test.txt", b"new content")

        self.assertFalse(result)
        self.assertEqual(file_path.read_bytes(), b"original")

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_atomic_write_new_file(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test atomic write creates new file."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        tm._atomic_write_secure_file("test.txt", b"content")

        file_path = Path(self.temp_dir) / "test.txt"
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.read_bytes(), b"content")
        self.assertEqual(file_path.stat().st_mode & 0o777, 0o600)

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_atomic_write_overwrites(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test atomic write overwrites existing file."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        file_path = Path(self.temp_dir) / "test.txt"
        file_path.write_bytes(b"original")

        tm._atomic_write_secure_file("test.txt", b"new content")

        self.assertEqual(file_path.read_bytes(), b"new content")

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_atomic_write_no_temp_file_on_success(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test that temp file is cleaned up after successful write."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        tm._atomic_write_secure_file("test.txt", b"content")

        # Check no temp files remain
        temp_files = list(Path(self.temp_dir).glob(".test.txt.*"))
        self.assertEqual(len(temp_files), 0)

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_read_secure_file_exists(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test reading existing file."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        file_path = Path(self.temp_dir) / "test.txt"
        file_path.write_bytes(b"content")

        result = tm._read_secure_file("test.txt")

        self.assertEqual(result, b"content")

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_read_secure_file_not_exists(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test reading non-existent file returns None."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        result = tm._read_secure_file("nonexistent.txt")

        self.assertIsNone(result)

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_delete_secure_file_exists(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test deleting existing file."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        file_path = Path(self.temp_dir) / "test.txt"
        file_path.write_bytes(b"content")

        tm._delete_secure_file("test.txt")

        self.assertFalse(file_path.exists())

    @patch("crewai.cli.shared.token_manager.TokenManager._get_or_create_key")
    def test_delete_secure_file_not_exists(
        self, mock_get_key: unittest.mock.MagicMock
    ) -> None:
        """Test deleting non-existent file doesn't raise."""
        mock_get_key.return_value = Fernet.generate_key()
        tm = TokenManager()

        # Should not raise
        tm._delete_secure_file("nonexistent.txt")


if __name__ == "__main__":
    unittest.main()
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai_cli.config import (
    CLI_SETTINGS_KEYS,
    DEFAULT_CLI_SETTINGS,
    USER_SETTINGS_KEYS,
    Settings,
)
from crewai_core.token_manager import TokenManager


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "settings.json"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_empty_initialization(self):
        settings = Settings(config_path=self.config_path)
        self.assertIsNone(settings.tool_repository_username)
        self.assertIsNone(settings.tool_repository_password)

    def test_initialization_with_data(self):
        settings = Settings(
            config_path=self.config_path, tool_repository_username="user1"
        )
        self.assertEqual(settings.tool_repository_username, "user1")
        self.assertIsNone(settings.tool_repository_password)

    def test_initialization_with_existing_file(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            json.dump({"tool_repository_username": "file_user"}, f)

        settings = Settings(config_path=self.config_path)
        self.assertEqual(settings.tool_repository_username, "file_user")

    def test_merge_file_and_input_data(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            json.dump(
                {
                    "tool_repository_username": "file_user",
                    "tool_repository_password": "file_pass",
                },
                f,
            )

        settings = Settings(
            config_path=self.config_path, tool_repository_username="new_user"
        )
        self.assertEqual(settings.tool_repository_username, "new_user")
        self.assertEqual(settings.tool_repository_password, "file_pass")

    def test_clear_user_settings(self):
        user_settings = {key: f"value_for_{key}" for key in USER_SETTINGS_KEYS}

        settings = Settings(config_path=self.config_path, **user_settings)
        settings.clear_user_settings()

        for key in user_settings.keys():
            self.assertEqual(getattr(settings, key), None)

    @patch("crewai_core.settings.TokenManager")
    def test_reset_settings(self, mock_token_manager):
        user_settings = {key: f"value_for_{key}" for key in USER_SETTINGS_KEYS}
        cli_settings = {key: f"value_for_{key}" for key in CLI_SETTINGS_KEYS if key != "oauth2_extra"}
        cli_settings["oauth2_extra"] = {"scope": "xxx", "other": "yyy"}

        settings = Settings(
            config_path=self.config_path, **user_settings, **cli_settings
        )

        mock_token_manager.return_value = MagicMock()
        TokenManager().save_tokens(
            "aaa.bbb.ccc", (datetime.now() + timedelta(seconds=36000)).timestamp()
        )

        settings.reset()

        for key in user_settings.keys():
            self.assertEqual(getattr(settings, key), None)
        for key in cli_settings.keys():
            self.assertEqual(getattr(settings, key), DEFAULT_CLI_SETTINGS.get(key))

        mock_token_manager.return_value.clear_tokens.assert_called_once()

    def test_dump_new_settings(self):
        settings = Settings(
            config_path=self.config_path, tool_repository_username="user1"
        )
        settings.dump()

        with self.config_path.open("r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["tool_repository_username"], "user1")

    def test_update_existing_settings(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            json.dump({"existing_setting": "value"}, f)

        settings = Settings(
            config_path=self.config_path, tool_repository_username="user1"
        )
        settings.dump()

        with self.config_path.open("r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["existing_setting"], "value")
        self.assertEqual(saved_data["tool_repository_username"], "user1")

    def test_none_values(self):
        settings = Settings(config_path=self.config_path, tool_repository_username=None)
        settings.dump()

        with self.config_path.open("r") as f:
            saved_data = json.load(f)

        self.assertIsNone(saved_data.get("tool_repository_username"))

    def test_invalid_json_in_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            f.write("invalid json")

        try:
            settings = Settings(config_path=self.config_path)
            self.assertIsNone(settings.tool_repository_username)
        except json.JSONDecodeError:
            self.fail("Settings initialization should handle invalid JSON")

    def test_empty_config_file(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.touch()

        settings = Settings(config_path=self.config_path)
        self.assertIsNone(settings.tool_repository_username)


class TestSettingsFilePermissions(unittest.TestCase):
    """Regression tests: credentials in settings.json must not be world-readable."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @unittest.skipIf(sys.platform == "win32", "POSIX permission semantics")
    def test_dump_writes_owner_only_file(self):
        config_path = self.test_dir / "settings.json"
        old_umask = os.umask(0o022)
        try:
            settings = Settings(
                config_path=config_path, tool_repository_password="hunter2"
            )
            settings.dump()
        finally:
            os.umask(old_umask)

        mode = stat.S_IMODE(config_path.stat().st_mode)
        self.assertEqual(mode, 0o600, f"expected 0o600, got {oct(mode)}")

    @unittest.skipIf(sys.platform == "win32", "POSIX permission semantics")
    def test_dedicated_config_dir_is_owner_only(self):
        config_path = self.test_dir / "crewai" / "settings.json"
        old_umask = os.umask(0o022)
        try:
            Settings(config_path=config_path, tool_repository_username="u")
        finally:
            os.umask(old_umask)

        mode = stat.S_IMODE(config_path.parent.stat().st_mode)
        self.assertEqual(mode, 0o700, f"expected 0o700, got {oct(mode)}")

    @unittest.skipIf(sys.platform == "win32", "POSIX permission semantics")
    def test_shared_fallback_dir_is_not_chmodded(self):
        """The system temp dir (a fallback parent) must never be globally chmod'd."""
        from crewai_core.settings import _ensure_dir_mode

        tmp_root = Path(tempfile.gettempdir())
        before = stat.S_IMODE(tmp_root.stat().st_mode)
        _ensure_dir_mode(tmp_root)
        after = stat.S_IMODE(tmp_root.stat().st_mode)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

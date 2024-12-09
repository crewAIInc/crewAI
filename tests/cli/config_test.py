import json
import shutil
import tempfile
import unittest
from pathlib import Path

from crewai.cli.config import Settings


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

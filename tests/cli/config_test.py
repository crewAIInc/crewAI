import json
import shutil
import tempfile
import unittest
from pathlib import Path

from crewai.cli.config import Settings


class TestSettings(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "settings.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_empty_initialization(self) -> None:
        settings = Settings(config_path=self.config_path)
        assert settings.tool_repository_username is None
        assert settings.tool_repository_password is None

    def test_initialization_with_data(self) -> None:
        settings = Settings(
            config_path=self.config_path, tool_repository_username="user1",
        )
        assert settings.tool_repository_username == "user1"
        assert settings.tool_repository_password is None

    def test_initialization_with_existing_file(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            json.dump({"tool_repository_username": "file_user"}, f)

        settings = Settings(config_path=self.config_path)
        assert settings.tool_repository_username == "file_user"

    def test_merge_file_and_input_data(self) -> None:
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
            config_path=self.config_path, tool_repository_username="new_user",
        )
        assert settings.tool_repository_username == "new_user"
        assert settings.tool_repository_password == "file_pass"

    def test_dump_new_settings(self) -> None:
        settings = Settings(
            config_path=self.config_path, tool_repository_username="user1",
        )
        settings.dump()

        with self.config_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data["tool_repository_username"] == "user1"

    def test_update_existing_settings(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            json.dump({"existing_setting": "value"}, f)

        settings = Settings(
            config_path=self.config_path, tool_repository_username="user1",
        )
        settings.dump()

        with self.config_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data["existing_setting"] == "value"
        assert saved_data["tool_repository_username"] == "user1"

    def test_none_values(self) -> None:
        settings = Settings(config_path=self.config_path, tool_repository_username=None)
        settings.dump()

        with self.config_path.open("r") as f:
            saved_data = json.load(f)

        assert saved_data.get("tool_repository_username") is None

    def test_invalid_json_in_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w") as f:
            f.write("invalid json")

        try:
            settings = Settings(config_path=self.config_path)
            assert settings.tool_repository_username is None
        except json.JSONDecodeError:
            self.fail("Settings initialization should handle invalid JSON")

    def test_empty_config_file(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.touch()

        settings = Settings(config_path=self.config_path)
        assert settings.tool_repository_username is None

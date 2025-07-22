import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from crewai.cli.settings.main import SettingsCommand
from crewai.cli.config import (
    Settings,
    USER_SETTINGS_KEYS,
    CLI_SETTINGS_KEYS,
    DEFAULT_CLI_SETTINGS,
    HIDDEN_SETTINGS_KEYS,
    READONLY_SETTINGS_KEYS,
)
import shutil


class TestSettingsCommand(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "settings.json"
        self.settings = Settings(config_path=self.config_path)
        self.settings_command = SettingsCommand(
            settings_kwargs={"config_path": self.config_path}
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("crewai.cli.settings.main.console")
    @patch("crewai.cli.settings.main.Table")
    def test_list_settings(self, mock_table_class, mock_console):
        mock_table_instance = MagicMock()
        mock_table_class.return_value = mock_table_instance

        self.settings_command.list()

        # Tests that the table is created skipping hidden settings
        mock_table_instance.add_row.assert_has_calls(
            [
                call(
                    field_name,
                    getattr(self.settings, field_name) or "Not set",
                    field_info.description,
                )
                for field_name, field_info in Settings.model_fields.items()
                if field_name not in HIDDEN_SETTINGS_KEYS
            ]
        )

        # Tests that the table is printed
        mock_console.print.assert_called_once_with(mock_table_instance)

    def test_set_valid_keys(self):
        valid_keys = Settings.model_fields.keys() - (
            READONLY_SETTINGS_KEYS + HIDDEN_SETTINGS_KEYS
        )
        for key in valid_keys:
            test_value = f"some_value_for_{key}"
            self.settings_command.set(key, test_value)
            self.assertEqual(getattr(self.settings_command.settings, key), test_value)

    def test_set_invalid_key(self):
        with self.assertRaises(SystemExit):
            self.settings_command.set("invalid_key", "value")

    def test_set_readonly_keys(self):
        for key in READONLY_SETTINGS_KEYS:
            with self.assertRaises(SystemExit):
                self.settings_command.set(key, "some_readonly_key_value")

    def test_set_hidden_keys(self):
        for key in HIDDEN_SETTINGS_KEYS:
            with self.assertRaises(SystemExit):
                self.settings_command.set(key, "some_hidden_key_value")

    def test_reset_all_settings(self):
        for key in USER_SETTINGS_KEYS + CLI_SETTINGS_KEYS:
            setattr(self.settings_command.settings, key, f"custom_value_for_{key}")
        self.settings_command.settings.dump()

        self.settings_command.reset_all_settings()

        print(USER_SETTINGS_KEYS)
        for key in USER_SETTINGS_KEYS:
            self.assertEqual(getattr(self.settings_command.settings, key), None)

        for key in CLI_SETTINGS_KEYS:
            self.assertEqual(
                getattr(self.settings_command.settings, key), DEFAULT_CLI_SETTINGS[key]
            )


#     @patch('crewai.cli.settings_command.console')
#     def test_list_settings(self, mock_console):
#         """Test listing all settings."""
#         self.settings_command.list()

#         # Verify that console.print was called with a table
#         mock_console.print.assert_called_once()

#     @patch('crewai.cli.settings_command.console')
#     def test_set_valid_key(self, mock_console):
#         """Test setting a valid configuration key."""
#         # Mock the settings instance
#         with patch.object(self.settings_command, 'settings') as mock_settings:
#             mock_settings.model_fields = {
#                 'enterprise_base_url': MagicMock(),
#                 'tool_repository_username': MagicMock(),
#                 'org_name': MagicMock()
#             }

#             self.settings_command.set('org_name', 'test_org')

#             # Verify the value was set and saved
#             mock_settings.__setattr__.assert_called_with('org_name', 'test_org')
#             mock_settings.dump.assert_called_once()
#             mock_console.print.assert_called_with(
#                 "Successfully set 'org_name' to 'test_org'",
#                 style="bold green"
#             )

#     @patch('crewai.cli.settings_command.console')
#     @patch('sys.exit')
#     def test_set_invalid_key(self, mock_exit, mock_console):
#         """Test setting an invalid configuration key."""
#         # Mock the settings instance with limited fields
#         with patch.object(self.settings_command, 'settings') as mock_settings:
#             mock_settings.model_fields = {
#                 'enterprise_base_url': MagicMock(),
#                 'tool_repository_username': MagicMock()
#             }

#             self.settings_command.set('invalid_key', 'value')

#             # Verify error message and exit
#             mock_console.print.assert_any_call(
#                 "Error: Unknown configuration key 'invalid_key'",
#                 style="bold red"
#             )
#             mock_exit.assert_called_with(1)

#     @patch('crewai.cli.settings_command.console')
#     def test_reset_all_settings(self, mock_console):
#         """Test resetting all settings to default values."""
#         # Mock the settings instance
#         with patch.object(self.settings_command, 'settings') as mock_settings:
#             self.settings_command.reset_all_settings()

#             # Verify reset was called
#             mock_settings.reset.assert_called_once()
#             mock_console.print.assert_called_with(
#                 "Successfully reset all configuration parameters to default values",
#                 style="bold green"
#             )

#     def test_settings_command_initialization(self):
#         """Test that SettingsCommand initializes correctly."""
#         self.assertIsInstance(self.settings_command.settings, Settings)
#         self.assertIsNotNone(self.settings_command.settings)


# if __name__ == '__main__':
#     unittest.main()

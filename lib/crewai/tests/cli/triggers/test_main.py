import json
import subprocess
import unittest
from unittest.mock import Mock, patch

import requests
from crewai.cli.triggers.main import TriggersCommand


class TestTriggersCommand(unittest.TestCase):
    @patch("crewai.cli.command.get_auth_token")
    @patch("crewai.cli.command.PlusAPI")
    def setUp(self, mock_plus_api, mock_get_auth_token):
        self.mock_get_auth_token = mock_get_auth_token
        self.mock_plus_api = mock_plus_api

        self.mock_get_auth_token.return_value = "test_token"

        self.triggers_command = TriggersCommand()
        self.mock_client = self.triggers_command.plus_api_client

    @patch("crewai.cli.triggers.main.console.print")
    def test_list_triggers_success(self, mock_console_print):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "apps": [
                {
                    "name": "Test App",
                    "slug": "test-app",
                    "description": "A test application",
                    "is_connected": True,
                    "triggers": [
                        {
                            "name": "Test Trigger",
                            "slug": "test-trigger",
                            "description": "A test trigger"
                        }
                    ]
                }
            ]
        }
        self.mock_client.get_triggers.return_value = mock_response

        self.triggers_command.list_triggers()

        self.mock_client.get_triggers.assert_called_once()
        mock_console_print.assert_any_call("[bold blue]Fetching available triggers...[/bold blue]")

    @patch("crewai.cli.triggers.main.console.print")
    def test_list_triggers_no_apps(self, mock_console_print):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {"apps": []}
        self.mock_client.get_triggers.return_value = mock_response

        self.triggers_command.list_triggers()

        mock_console_print.assert_any_call("[yellow]No triggers found.[/yellow]")

    @patch("crewai.cli.triggers.main.console.print")
    def test_list_triggers_api_error(self, mock_console_print):
        self.mock_client.get_triggers.side_effect = Exception("API Error")

        with self.assertRaises(SystemExit):
            self.triggers_command.list_triggers()

        mock_console_print.assert_any_call("[bold red]Error fetching triggers: API Error[/bold red]")

    @patch("crewai.cli.triggers.main.console.print")
    def test_execute_with_trigger_invalid_format(self, mock_console_print):
        with self.assertRaises(SystemExit):
            self.triggers_command.execute_with_trigger("invalid-format")

        mock_console_print.assert_called_with(
            "[bold red]Error: Trigger must be in format 'app_slug/trigger_slug'[/bold red]"
        )

    @patch("crewai.cli.triggers.main.console.print")
    @patch.object(TriggersCommand, "_run_crew_with_payload")
    def test_execute_with_trigger_success(self, mock_run_crew, mock_console_print):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.ok = True
        mock_response.json.return_value = {
            "sample_payload": {"key": "value", "data": "test"}
        }
        self.mock_client.get_trigger_payload.return_value = mock_response

        self.triggers_command.execute_with_trigger("test-app/test-trigger")

        self.mock_client.get_trigger_payload.assert_called_once_with("test-app", "test-trigger")
        mock_run_crew.assert_called_once_with({"key": "value", "data": "test"})
        mock_console_print.assert_any_call(
            "[bold blue]Fetching trigger payload for test-app/test-trigger...[/bold blue]"
        )

    @patch("crewai.cli.triggers.main.console.print")
    def test_execute_with_trigger_not_found(self, mock_console_print):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Trigger not found"}
        self.mock_client.get_trigger_payload.return_value = mock_response

        with self.assertRaises(SystemExit):
            self.triggers_command.execute_with_trigger("test-app/nonexistent-trigger")

        mock_console_print.assert_any_call("[bold red]Error: Trigger not found[/bold red]")

    @patch("crewai.cli.triggers.main.console.print")
    def test_execute_with_trigger_api_error(self, mock_console_print):
        self.mock_client.get_trigger_payload.side_effect = Exception("API Error")

        with self.assertRaises(SystemExit):
            self.triggers_command.execute_with_trigger("test-app/test-trigger")

        mock_console_print.assert_any_call(
            "[bold red]Error executing crew with trigger: API Error[/bold red]"
        )


    @patch("subprocess.run")
    def test_run_crew_with_payload_success(self, mock_subprocess):
        payload = {"key": "value", "data": "test"}
        mock_subprocess.return_value = None

        self.triggers_command._run_crew_with_payload(payload)

        mock_subprocess.assert_called_once_with(
            ["uv", "run", "run_with_trigger", json.dumps(payload)],
            capture_output=False,
            text=True,
            check=True
        )

    @patch("subprocess.run")
    def test_run_crew_with_payload_failure(self, mock_subprocess):
        payload = {"key": "value"}
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "uv")

        with self.assertRaises(SystemExit):
            self.triggers_command._run_crew_with_payload(payload)

    @patch("subprocess.run")
    def test_run_crew_with_payload_empty_payload(self, mock_subprocess):
        payload = {}
        mock_subprocess.return_value = None

        self.triggers_command._run_crew_with_payload(payload)

        mock_subprocess.assert_called_once_with(
            ["uv", "run", "run_with_trigger", "{}"],
            capture_output=False,
            text=True,
            check=True
        )

    @patch("crewai.cli.triggers.main.console.print")
    def test_execute_with_trigger_with_default_error_message(self, mock_console_print):
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {}
        self.mock_client.get_trigger_payload.return_value = mock_response

        with self.assertRaises(SystemExit):
            self.triggers_command.execute_with_trigger("test-app/test-trigger")

        mock_console_print.assert_any_call("[bold red]Error: Trigger not found[/bold red]")

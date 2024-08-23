import unittest
from unittest.mock import MagicMock, patch

import requests
from crewai.cli.authentication.main import AuthenticationCommand


class TestAuthenticationCommand(unittest.TestCase):
    def setUp(self):
        self.auth_command = AuthenticationCommand()

    @patch("crewai.cli.authentication.main.requests.post")
    def test_get_device_code(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "device_code": "123456",
            "user_code": "ABCDEF",
            "verification_uri_complete": "https://example.com",
            "interval": 5,
        }
        mock_post.return_value = mock_response

        device_code_data = self.auth_command._get_device_code()

        self.assertEqual(device_code_data["device_code"], "123456")
        self.assertEqual(device_code_data["user_code"], "ABCDEF")
        self.assertEqual(
            device_code_data["verification_uri_complete"], "https://example.com"
        )
        self.assertEqual(device_code_data["interval"], 5)

    @patch("crewai.cli.authentication.main.console.print")
    @patch("crewai.cli.authentication.main.webbrowser.open")
    def test_display_auth_instructions(self, mock_open, mock_print):
        device_code_data = {
            "verification_uri_complete": "https://example.com",
            "user_code": "ABCDEF",
        }

        self.auth_command._display_auth_instructions(device_code_data)

        mock_print.assert_any_call("1. Navigate to: ", "https://example.com")
        mock_print.assert_any_call("2. Enter the following code: ", "ABCDEF")
        mock_open.assert_called_once_with("https://example.com")

    @patch("crewai.cli.authentication.main.requests.post")
    @patch("crewai.cli.authentication.main.validate_token")
    @patch("crewai.cli.authentication.main.console.print")
    def test_poll_for_token_success(self, mock_print, mock_validate_token, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id_token": "TOKEN",
            "access_token": "ACCESS_TOKEN",
        }
        mock_post.return_value = mock_response

        self.auth_command._poll_for_token({"device_code": "123456"})

        mock_validate_token.assert_called_once_with("TOKEN")
        mock_print.assert_called_once_with("\nWelcome to CrewAI+ !!", style="green")

    @patch("crewai.cli.authentication.main.requests.post")
    @patch("crewai.cli.authentication.main.console.print")
    def test_poll_for_token_error(self, mock_print, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "invalid_request",
            "error_description": "Invalid request",
        }
        mock_post.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            self.auth_command._poll_for_token({"device_code": "123456"})

        mock_print.assert_not_called()

    @patch("crewai.cli.authentication.main.requests.post")
    @patch("crewai.cli.authentication.main.console.print")
    def test_poll_for_token_timeout(self, mock_print, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": "authorization_pending",
            "error_description": "Authorization pending",
        }
        mock_post.return_value = mock_response

        self.auth_command._poll_for_token({"device_code": "123456", "interval": 0.01})

        mock_print.assert_called_once_with(
            "Timeout: Failed to get the token. Please try again.", style="bold red"
        )

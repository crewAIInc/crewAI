from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pytest
import requests
from crewai.cli.authentication.main import AuthenticationCommand
from crewai.cli.constants import (
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
    CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
)


class TestAuthenticationCommand:
    def setup_method(self):
        self.auth_command = AuthenticationCommand()

    # TODO: these expectations are reading from the actual settings, we should mock them.
    # E.g. if you change the client_id locally, this test will fail.
    @pytest.mark.parametrize(
        "user_provider,expected_urls",
        [
            (
                "workos",
                {
                    "device_code_url": f"https://{CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN}/oauth2/device_authorization",
                    "token_url": f"https://{CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN}/oauth2/token",
                    "client_id": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID,
                    "audience": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
                    "domain": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
                },
            ),
        ],
    )
    @patch("crewai.cli.authentication.main.AuthenticationCommand._get_device_code")
    @patch(
        "crewai.cli.authentication.main.AuthenticationCommand._display_auth_instructions"
    )
    @patch("crewai.cli.authentication.main.AuthenticationCommand._poll_for_token")
    @patch("crewai.cli.authentication.main.console.print")
    def test_login(
        self,
        mock_console_print,
        mock_poll,
        mock_display,
        mock_get_device,
        user_provider,
        expected_urls,
    ):
        mock_get_device.return_value = {
            "device_code": "test_code",
            "user_code": "123456",
        }

        self.auth_command.login()

        mock_console_print.assert_called_once_with(
            "Signing in to CrewAI AOP...\n", style="bold blue"
        )
        mock_get_device.assert_called_once()
        mock_display.assert_called_once_with(
            {"device_code": "test_code", "user_code": "123456"}
        )
        mock_poll.assert_called_once_with(
            {"device_code": "test_code", "user_code": "123456"},
        )
        assert (
            self.auth_command.oauth2_provider.get_client_id()
            == expected_urls["client_id"]
        )
        assert (
            self.auth_command.oauth2_provider.get_audience()
            == expected_urls["audience"]
        )
        assert (
            self.auth_command.oauth2_provider._get_domain() == expected_urls["domain"]
        )

    @patch("crewai.cli.authentication.main.webbrowser")
    @patch("crewai.cli.authentication.main.console.print")
    def test_display_auth_instructions(self, mock_console_print, mock_webbrowser):
        device_code_data = {
            "verification_uri_complete": "https://example.com/auth",
            "user_code": "123456",
        }

        self.auth_command._display_auth_instructions(device_code_data)

        expected_calls = [
            call("1. Navigate to: ", "https://example.com/auth"),
            call("2. Enter the following code: ", "123456"),
        ]
        mock_console_print.assert_has_calls(expected_calls)
        mock_webbrowser.open.assert_called_once_with("https://example.com/auth")

    @pytest.mark.parametrize(
        "user_provider,jwt_config",
        [
            (
                "workos",
                {
                    "jwks_url": f"https://{CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN}/oauth2/jwks",
                    "issuer": f"https://{CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN}",
                    "audience": CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE,
                },
            ),
        ],
    )
    @pytest.mark.parametrize("has_expiration", [True, False])
    @patch("crewai.cli.authentication.main.validate_jwt_token")
    @patch("crewai.cli.authentication.main.TokenManager.save_tokens")
    def test_validate_and_save_token(
        self,
        mock_save_tokens,
        mock_validate_jwt,
        user_provider,
        jwt_config,
        has_expiration,
    ):
        from crewai.cli.authentication.main import Oauth2Settings
        from crewai.cli.authentication.providers.workos import WorkosProvider

        if user_provider == "workos":
            self.auth_command.oauth2_provider = WorkosProvider(
                settings=Oauth2Settings(
                    provider=user_provider,
                    client_id="test-client-id",
                    domain=CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN,
                    audience=jwt_config["audience"],
                )
            )

        token_data = {"access_token": "test_access_token", "id_token": "test_id_token"}

        if has_expiration:
            future_timestamp = int((datetime.now() + timedelta(days=100)).timestamp())
            decoded_token = {"exp": future_timestamp}
        else:
            decoded_token = {}

        mock_validate_jwt.return_value = decoded_token

        self.auth_command._validate_and_save_token(token_data)

        mock_validate_jwt.assert_called_once_with(
            jwt_token="test_access_token",
            jwks_url=jwt_config["jwks_url"],
            issuer=jwt_config["issuer"],
            audience=jwt_config["audience"],
        )

        if has_expiration:
            mock_save_tokens.assert_called_once_with(
                "test_access_token", future_timestamp
            )
        else:
            mock_save_tokens.assert_called_once_with("test_access_token", 0)

    @patch("crewai.cli.tools.main.ToolCommand")
    @patch("crewai.cli.authentication.main.Settings")
    @patch("crewai.cli.authentication.main.console.print")
    def test_login_to_tool_repository_success(
        self, mock_console_print, mock_settings, mock_tool_command
    ):
        mock_tool_instance = MagicMock()
        mock_tool_command.return_value = mock_tool_instance

        mock_settings_instance = MagicMock()
        mock_settings_instance.org_name = "Test Org"
        mock_settings_instance.org_uuid = "test-uuid-123"
        mock_settings.return_value = mock_settings_instance

        self.auth_command._login_to_tool_repository()

        mock_tool_command.assert_called_once()
        mock_tool_instance.login.assert_called_once()

        expected_calls = [
            call(
                "Now logging you in to the Tool Repository... ",
                style="bold blue",
                end="",
            ),
            call("Success!\n", style="bold green"),
            call(
                "You are now authenticated to the tool repository for organization [bold cyan]'Test Org'[/bold cyan]",
                style="green",
            ),
        ]
        mock_console_print.assert_has_calls(expected_calls)

    @patch("crewai.cli.tools.main.ToolCommand")
    @patch("crewai.cli.authentication.main.console.print")
    def test_login_to_tool_repository_error(
        self, mock_console_print, mock_tool_command
    ):
        mock_tool_instance = MagicMock()
        mock_tool_instance.login.side_effect = Exception("Tool repository error")
        mock_tool_command.return_value = mock_tool_instance

        self.auth_command._login_to_tool_repository()

        mock_tool_command.assert_called_once()
        mock_tool_instance.login.assert_called_once()

        expected_calls = [
            call(
                "Now logging you in to the Tool Repository... ",
                style="bold blue",
                end="",
            ),
            call(
                "\n[bold yellow]Warning:[/bold yellow] Authentication with the Tool Repository failed.",
                style="yellow",
            ),
            call(
                "Other features will work normally, but you may experience limitations with downloading and publishing tools.\nRun [bold]crewai login[/bold] to try logging in again.\n",
                style="yellow",
            ),
        ]
        mock_console_print.assert_has_calls(expected_calls)

    @patch("requests.post")
    def test_get_device_code(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "123456",
            "verification_uri_complete": "https://example.com/auth",
        }
        mock_post.return_value = mock_response

        self.auth_command.oauth2_provider = MagicMock()
        self.auth_command.oauth2_provider.get_client_id.return_value = "test_client"
        self.auth_command.oauth2_provider.get_authorize_url.return_value = (
            "https://example.com/device"
        )
        self.auth_command.oauth2_provider.get_audience.return_value = "test_audience"
        self.auth_command.oauth2_provider.get_oauth_scopes.return_value = ["openid", "profile", "email"]

        result = self.auth_command._get_device_code()

        mock_post.assert_called_once_with(
            url="https://example.com/device",
            data={
                "client_id": "test_client",
                "scope": "openid profile email",
                "audience": "test_audience",
            },
            timeout=20,
        )

        assert result == {
            "device_code": "test_device_code",
            "user_code": "123456",
            "verification_uri_complete": "https://example.com/auth",
        }

    @patch("requests.post")
    @patch("crewai.cli.authentication.main.console.print")
    def test_poll_for_token_success(self, mock_console_print, mock_post):
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "access_token": "test_access_token",
            "id_token": "test_id_token",
        }
        mock_post.return_value = mock_response_success

        device_code_data = {"device_code": "test_device_code", "interval": 1}

        with (
            patch.object(
                self.auth_command, "_validate_and_save_token"
            ) as mock_validate,
            patch.object(
                self.auth_command, "_login_to_tool_repository"
            ) as mock_tool_login,
        ):
            self.auth_command.oauth2_provider = MagicMock()
            self.auth_command.oauth2_provider.get_token_url.return_value = (
                "https://example.com/token"
            )
            self.auth_command.oauth2_provider.get_client_id.return_value = "test_client"

            self.auth_command._poll_for_token(device_code_data)

            mock_post.assert_called_once_with(
                "https://example.com/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": "test_device_code",
                    "client_id": "test_client",
                },
                timeout=30,
            )

            mock_validate.assert_called_once()
            mock_tool_login.assert_called_once()

            expected_calls = [
                call("\nWaiting for authentication... ", style="bold blue", end=""),
                call("Success!", style="bold green"),
                call("\n[bold green]Welcome to CrewAI AOP![/bold green]\n"),
            ]
            mock_console_print.assert_has_calls(expected_calls)

    @patch("requests.post")
    @patch("crewai.cli.authentication.main.console.print")
    def test_poll_for_token_timeout(self, mock_console_print, mock_post):
        mock_response_pending = MagicMock()
        mock_response_pending.status_code = 400
        mock_response_pending.json.return_value = {"error": "authorization_pending"}
        mock_post.return_value = mock_response_pending

        device_code_data = {
            "device_code": "test_device_code",
            "interval": 0.1,  # Short interval for testing
        }

        self.auth_command._poll_for_token(device_code_data)

        mock_console_print.assert_any_call(
            "Timeout: Failed to get the token. Please try again.", style="bold red"
        )

    @patch("requests.post")
    def test_poll_for_token_error(self, mock_post):
        """Test the method to poll for token (error path)."""
        # Setup mock to return error
        mock_response_error = MagicMock()
        mock_response_error.status_code = 400
        mock_response_error.json.return_value = {
            "error": "access_denied",
            "error_description": "User denied access",
        }
        mock_post.return_value = mock_response_error

        device_code_data = {"device_code": "test_device_code", "interval": 1}

        with pytest.raises(requests.HTTPError):
            self.auth_command._poll_for_token(device_code_data)

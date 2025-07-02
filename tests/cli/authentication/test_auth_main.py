import pytest
from datetime import datetime, timedelta
import requests
from unittest.mock import MagicMock, patch, call
from crewai.cli.authentication.main import AuthenticationCommand
from crewai.cli.authentication.constants import (
    AUTH0_AUDIENCE,
    AUTH0_CLIENT_ID,
    AUTH0_DOMAIN,
    WORKOS_DOMAIN,
    WORKOS_CLI_CONNECT_APP_ID,
    WORKOS_ENVIRONMENT_ID,
)


class TestAuthenticationCommand:
    def setup_method(self):
        self.auth_command = AuthenticationCommand()

    @pytest.mark.parametrize(
        "user_provider,expected_urls",
        [
            (
                "auth0",
                {
                    "device_code_url": f"https://{AUTH0_DOMAIN}/oauth/device/code",
                    "token_url": f"https://{AUTH0_DOMAIN}/oauth/token",
                    "client_id": AUTH0_CLIENT_ID,
                    "audience": AUTH0_AUDIENCE,
                },
            ),
            (
                "workos",
                {
                    "device_code_url": f"https://{WORKOS_DOMAIN}/oauth2/device_authorization",
                    "token_url": f"https://{WORKOS_DOMAIN}/oauth2/token",
                    "client_id": WORKOS_CLI_CONNECT_APP_ID,
                },
            ),
        ],
    )
    @patch(
        "crewai.cli.authentication.main.AuthenticationCommand._determine_user_provider"
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
        mock_determine_provider,
        user_provider,
        expected_urls,
    ):
        mock_determine_provider.return_value = user_provider
        mock_get_device.return_value = {
            "device_code": "test_code",
            "user_code": "123456",
        }

        self.auth_command.login()

        mock_console_print.assert_called_once_with(
            "Signing in to CrewAI Enterprise...\n", style="bold blue"
        )
        mock_determine_provider.assert_called_once()
        mock_get_device.assert_called_once_with(
            expected_urls["client_id"],
            expected_urls["device_code_url"],
            expected_urls.get("audience", None),
        )
        mock_display.assert_called_once_with(
            {"device_code": "test_code", "user_code": "123456"}
        )
        mock_poll.assert_called_once_with(
            {"device_code": "test_code", "user_code": "123456"},
            expected_urls["client_id"],
            expected_urls["token_url"],
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
                "auth0",
                {
                    "jwks_url": f"https://{AUTH0_DOMAIN}/.well-known/jwks.json",
                    "issuer": f"https://{AUTH0_DOMAIN}/",
                    "audience": AUTH0_AUDIENCE,
                },
            ),
            (
                "workos",
                {
                    "jwks_url": f"https://{WORKOS_DOMAIN}/oauth2/jwks",
                    "issuer": f"https://{WORKOS_DOMAIN}",
                    "audience": WORKOS_ENVIRONMENT_ID,
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
        self.auth_command.user_provider = user_provider
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
                "You are authenticated to the tool repository as [bold cyan]'Test Org'[/bold cyan] (test-uuid-123)",
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

    @pytest.mark.parametrize(
        "api_response,expected_provider",
        [
            ({"provider": "auth0"}, "auth0"),
            ({"provider": "workos"}, "workos"),
            ({"provider": "none"}, "workos"),  # Default to workos for any other value
            (
                {},
                "workos",
            ),  # Default to workos if no provider key is sent in the response
        ],
    )
    @patch("crewai.cli.authentication.main.PlusAPI")
    @patch("crewai.cli.authentication.main.console.print")
    @patch("builtins.input", return_value="test@example.com")
    def test_determine_user_provider_success(
        self,
        mock_input,
        mock_console_print,
        mock_plus_api,
        api_response,
        expected_provider,
    ):
        mock_api_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = api_response
        mock_api_instance._make_request.return_value = mock_response
        mock_plus_api.return_value = mock_api_instance

        result = self.auth_command._determine_user_provider()

        mock_input.assert_called_once()

        mock_plus_api.assert_called_once_with("")
        mock_api_instance._make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/me/provider?email=test%40example.com"
        )

        assert result == expected_provider

    @patch("crewai.cli.authentication.main.PlusAPI")
    @patch("crewai.cli.authentication.main.console.print")
    @patch("builtins.input", return_value="test@example.com")
    def test_determine_user_provider_error(
        self, mock_input, mock_console_print, mock_plus_api
    ):
        mock_api_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_api_instance._make_request.return_value = mock_response
        mock_plus_api.return_value = mock_api_instance

        with pytest.raises(SystemExit):
            self.auth_command._determine_user_provider()

        mock_input.assert_called_once()

        mock_plus_api.assert_called_once_with("")
        mock_api_instance._make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/me/provider?email=test%40example.com"
        )

        mock_console_print.assert_has_calls(
            [
                call(
                    "Enter your CrewAI Enterprise account email: ",
                    style="bold blue",
                    end="",
                ),
                call(
                    "Error: Failed to authenticate with crewai enterprise. Ensure that you are using the latest crewai version and please try again. If the problem persists, contact support@crewai.com.",
                    style="red",
                ),
            ]
        )

    @patch("requests.post")
    def test_get_device_code(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "device_code": "test_device_code",
            "user_code": "123456",
            "verification_uri_complete": "https://example.com/auth",
        }
        mock_post.return_value = mock_response

        result = self.auth_command._get_device_code(
            client_id="test_client",
            device_code_url="https://example.com/device",
            audience="test_audience",
        )

        mock_post.assert_called_once_with(
            url="https://example.com/device",
            data={
                "client_id": "test_client",
                "scope": "openid",
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
            self.auth_command._poll_for_token(
                device_code_data, "test_client", "https://example.com/token"
            )

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
                call("\n[bold green]Welcome to CrewAI Enterprise![/bold green]\n"),
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

        self.auth_command._poll_for_token(
            device_code_data, "test_client", "https://example.com/token"
        )

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
            self.auth_command._poll_for_token(
                device_code_data, "test_client", "https://example.com/token"
            )

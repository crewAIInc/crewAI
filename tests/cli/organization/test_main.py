import unittest
from unittest.mock import MagicMock, patch, call

import pytest
from click.testing import CliRunner
import requests

from crewai.cli.organization.main import OrganizationCommand
from crewai.cli.cli import list, switch, current


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def org_command():
    with patch.object(OrganizationCommand, '__init__', return_value=None):
        command = OrganizationCommand()
        yield command


@pytest.fixture
def mock_settings():
    with patch('crewai.cli.organization.main.Settings') as mock_settings_class:
        mock_settings_instance = MagicMock()
        mock_settings_class.return_value = mock_settings_instance
        yield mock_settings_instance


@patch('crewai.cli.cli.OrganizationCommand')
def test_org_list_command(mock_org_command_class, runner):
    mock_org_instance = MagicMock()
    mock_org_command_class.return_value = mock_org_instance

    result = runner.invoke(list)

    assert result.exit_code == 0
    mock_org_command_class.assert_called_once()
    mock_org_instance.list.assert_called_once()


@patch('crewai.cli.cli.OrganizationCommand')
def test_org_switch_command(mock_org_command_class, runner):
    mock_org_instance = MagicMock()
    mock_org_command_class.return_value = mock_org_instance

    result = runner.invoke(switch, ['test-id'])

    assert result.exit_code == 0
    mock_org_command_class.assert_called_once()
    mock_org_instance.switch.assert_called_once_with('test-id')


@patch('crewai.cli.cli.OrganizationCommand')
def test_org_current_command(mock_org_command_class, runner):
    mock_org_instance = MagicMock()
    mock_org_command_class.return_value = mock_org_instance

    result = runner.invoke(current)

    assert result.exit_code == 0
    mock_org_command_class.assert_called_once()
    mock_org_instance.current.assert_called_once()


class TestOrganizationCommand(unittest.TestCase):
    def setUp(self):
        with patch.object(OrganizationCommand, '__init__', return_value=None):
            self.org_command = OrganizationCommand()
            self.org_command.plus_api_client = MagicMock()

    @patch('crewai.cli.organization.main.console')
    @patch('crewai.cli.organization.main.Table')
    def test_list_organizations_success(self, mock_table, mock_console):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"name": "Org 1", "uuid": "org-123"},
            {"name": "Org 2", "uuid": "org-456"}
        ]
        self.org_command.plus_api_client = MagicMock()
        self.org_command.plus_api_client.get_organizations.return_value = mock_response

        mock_console.print = MagicMock()

        self.org_command.list()

        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_table.assert_called_once_with(title="Your Organizations")
        mock_table.return_value.add_column.assert_has_calls([
            call("Name", style="cyan"),
            call("ID", style="green")
        ])
        mock_table.return_value.add_row.assert_has_calls([
            call("Org 1", "org-123"),
            call("Org 2", "org-456")
        ])

    @patch('crewai.cli.organization.main.console')
    def test_list_organizations_empty(self, mock_console):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = []
        self.org_command.plus_api_client = MagicMock()
        self.org_command.plus_api_client.get_organizations.return_value = mock_response

        self.org_command.list()

        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_console.print.assert_called_once_with(
            "You don't belong to any organizations yet.",
            style="yellow"
        )

    @patch('crewai.cli.organization.main.console')
    def test_list_organizations_api_error(self, mock_console):
        self.org_command.plus_api_client = MagicMock()
        self.org_command.plus_api_client.get_organizations.side_effect = requests.exceptions.RequestException("API Error")

        with pytest.raises(SystemExit):
            self.org_command.list()


        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_console.print.assert_called_once_with(
            "Failed to retrieve organization list: API Error",
            style="bold red"
        )

    @patch('crewai.cli.organization.main.console')
    @patch('crewai.cli.organization.main.Settings')
    def test_switch_organization_success(self, mock_settings_class, mock_console):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"name": "Org 1", "uuid": "org-123"},
            {"name": "Test Org", "uuid": "test-id"}
        ]
        self.org_command.plus_api_client = MagicMock()
        self.org_command.plus_api_client.get_organizations.return_value = mock_response

        mock_settings_instance = MagicMock()
        mock_settings_class.return_value = mock_settings_instance

        self.org_command.switch("test-id")

        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_settings_instance.dump.assert_called_once()
        assert mock_settings_instance.org_name == "Test Org"
        assert mock_settings_instance.org_uuid == "test-id"
        mock_console.print.assert_called_once_with(
            "Successfully switched to Test Org (test-id)",
            style="bold green"
        )

    @patch('crewai.cli.organization.main.console')
    def test_switch_organization_not_found(self, mock_console):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {"name": "Org 1", "uuid": "org-123"},
            {"name": "Org 2", "uuid": "org-456"}
        ]
        self.org_command.plus_api_client = MagicMock()
        self.org_command.plus_api_client.get_organizations.return_value = mock_response

        self.org_command.switch("non-existent-id")

        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_console.print.assert_called_once_with(
            "Organization with id 'non-existent-id' not found.",
            style="bold red"
        )

    @patch('crewai.cli.organization.main.console')
    @patch('crewai.cli.organization.main.Settings')
    def test_current_organization_with_org(self, mock_settings_class, mock_console):
        mock_settings_instance = MagicMock()
        mock_settings_instance.org_name = "Test Org"
        mock_settings_instance.org_uuid = "test-id"
        mock_settings_class.return_value = mock_settings_instance

        self.org_command.current()

        self.org_command.plus_api_client.get_organizations.assert_not_called()
        mock_console.print.assert_called_once_with(
            "Currently logged in to organization Test Org (test-id)",
            style="bold green"
        )

    @patch('crewai.cli.organization.main.console')
    @patch('crewai.cli.organization.main.Settings')
    def test_current_organization_without_org(self, mock_settings_class, mock_console):
        mock_settings_instance = MagicMock()
        mock_settings_instance.org_uuid = None
        mock_settings_class.return_value = mock_settings_instance

        self.org_command.current()

        assert mock_console.print.call_count == 3
        mock_console.print.assert_any_call(
            "You're not currently logged in to any organization.",
            style="yellow"
        )

    @patch('crewai.cli.organization.main.console')
    def test_list_organizations_unauthorized(self, mock_console):
        mock_response = MagicMock()
        mock_http_error = requests.exceptions.HTTPError(
            "401 Client Error: Unauthorized",
            response=MagicMock(status_code=401)
        )

        mock_response.raise_for_status.side_effect = mock_http_error
        self.org_command.plus_api_client.get_organizations.return_value = mock_response

        self.org_command.list()

        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_console.print.assert_called_once_with(
            "You are not logged in to any organization. Use 'crewai login' to login.",
            style="bold red"
        )

    @patch('crewai.cli.organization.main.console')
    def test_switch_organization_unauthorized(self, mock_console):
        mock_response = MagicMock()
        mock_http_error = requests.exceptions.HTTPError(
            "401 Client Error: Unauthorized",
            response=MagicMock(status_code=401)
        )

        mock_response.raise_for_status.side_effect = mock_http_error
        self.org_command.plus_api_client.get_organizations.return_value = mock_response

        self.org_command.switch("test-id")

        self.org_command.plus_api_client.get_organizations.assert_called_once()
        mock_console.print.assert_called_once_with(
            "You are not logged in to any organization. Use 'crewai login' to login.",
            style="bold red"
        )

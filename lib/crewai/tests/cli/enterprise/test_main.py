import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from requests.exceptions import JSONDecodeError

from crewai.cli.enterprise.main import EnterpriseConfigureCommand
from crewai.cli.settings.main import SettingsCommand
import shutil


class TestEnterpriseConfigureCommand(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_path = self.test_dir / "settings.json"

        with patch('crewai.cli.enterprise.main.SettingsCommand') as mock_settings_command_class:
            self.mock_settings_command = Mock(spec=SettingsCommand)
            mock_settings_command_class.return_value = self.mock_settings_command

            self.enterprise_command = EnterpriseConfigureCommand()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('crewai.cli.enterprise.main.requests.get')
    @patch('crewai.cli.enterprise.main.get_crewai_version')
    def test_successful_configuration(self, mock_get_version, mock_requests_get):
        mock_get_version.return_value = "1.0.0"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'audience': 'test_audience',
            'domain': 'test.domain.com',
            'device_authorization_client_id': 'test_client_id',
            'provider': 'workos',
            'extra': {}
        }
        mock_requests_get.return_value = mock_response

        enterprise_url = "https://enterprise.example.com"
        self.enterprise_command.configure(enterprise_url)

        expected_headers = {
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-CLI/1.0.0",
            "X-Crewai-Version": "1.0.0",
        }
        mock_requests_get.assert_called_once_with(
            "https://enterprise.example.com/auth/parameters",
            timeout=30,
            headers=expected_headers
        )

        expected_calls = [
            ('enterprise_base_url', 'https://enterprise.example.com'),
            ('oauth2_provider', 'workos'),
            ('oauth2_audience', 'test_audience'),
            ('oauth2_client_id', 'test_client_id'),
            ('oauth2_domain', 'test.domain.com'),
            ('oauth2_extra', {})
        ]

        actual_calls = self.mock_settings_command.set.call_args_list
        self.assertEqual(len(actual_calls), 6)

        for i, (key, value) in enumerate(expected_calls):
            call_args = actual_calls[i][0]
            self.assertEqual(call_args[0], key)
            self.assertEqual(call_args[1], value)

    @patch('crewai.cli.enterprise.main.requests.get')
    @patch('crewai.cli.enterprise.main.get_crewai_version')
    def test_http_error_handling(self, mock_get_version, mock_requests_get):
        mock_get_version.return_value = "1.0.0"

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_requests_get.return_value = mock_response

        with self.assertRaises(SystemExit):
            self.enterprise_command.configure("https://enterprise.example.com")

    @patch('crewai.cli.enterprise.main.requests.get')
    @patch('crewai.cli.enterprise.main.get_crewai_version')
    def test_invalid_json_response(self, mock_get_version, mock_requests_get):
        mock_get_version.return_value = "1.0.0"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = JSONDecodeError("Invalid JSON", "", 0)
        mock_requests_get.return_value = mock_response

        with self.assertRaises(SystemExit):
            self.enterprise_command.configure("https://enterprise.example.com")

    @patch('crewai.cli.enterprise.main.requests.get')
    @patch('crewai.cli.enterprise.main.get_crewai_version')
    def test_missing_required_fields(self, mock_get_version, mock_requests_get):
        mock_get_version.return_value = "1.0.0"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'audience': 'test_audience',
        }
        mock_requests_get.return_value = mock_response

        with self.assertRaises(SystemExit):
            self.enterprise_command.configure("https://enterprise.example.com")

    @patch('crewai.cli.enterprise.main.requests.get')
    @patch('crewai.cli.enterprise.main.get_crewai_version')
    def test_settings_update_error(self, mock_get_version, mock_requests_get):
        mock_get_version.return_value = "1.0.0"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'audience': 'test_audience',
            'domain': 'test.domain.com',
            'device_authorization_client_id': 'test_client_id',
            'provider': 'workos'
        }
        mock_requests_get.return_value = mock_response

        self.mock_settings_command.set.side_effect = Exception("Settings update failed")

        with self.assertRaises(SystemExit):
            self.enterprise_command.configure("https://enterprise.example.com")

    def test_url_trailing_slash_removal(self):
        with patch.object(self.enterprise_command, '_fetch_oauth_config') as mock_fetch, \
             patch.object(self.enterprise_command, '_update_oauth_settings') as mock_update:

            mock_fetch.return_value = {
                'audience': 'test_audience',
                'domain': 'test.domain.com',
                'device_authorization_client_id': 'test_client_id',
                'provider': 'workos'
            }

            self.enterprise_command.configure("https://enterprise.example.com/")

            mock_fetch.assert_called_once_with("https://enterprise.example.com")
            mock_update.assert_called_once()

import unittest
from os import environ
from unittest.mock import MagicMock, patch

from crewai.cli.deploy.api import CrewAPI


class TestCrewAPI(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api = CrewAPI(self.api_key)

    def test_init(self):
        self.assertEqual(self.api.api_key, self.api_key)
        self.assertEqual(
            self.api.headers,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CrewAI-CLI/no-version-found"
            },
        )

    @patch("crewai.cli.deploy.api.requests.request")
    def test_make_request(self, mock_request):
        mock_response = MagicMock()
        mock_request.return_value = mock_response

        response = self.api._make_request("GET", "test_endpoint")

        mock_request.assert_called_once_with(
            "GET", f"{self.api.base_url}/test_endpoint", headers=self.api.headers
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_deploy_by_name(self, mock_make_request):
        self.api.deploy_by_name("test_project")
        mock_make_request.assert_called_once_with("POST", "by-name/test_project/deploy")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_deploy_by_uuid(self, mock_make_request):
        self.api.deploy_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with("POST", "test_uuid/deploy")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_status_by_name(self, mock_make_request):
        self.api.status_by_name("test_project")
        mock_make_request.assert_called_once_with("GET", "by-name/test_project/status")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_status_by_uuid(self, mock_make_request):
        self.api.status_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with("GET", "test_uuid/status")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_logs_by_name(self, mock_make_request):
        self.api.logs_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "GET", "by-name/test_project/logs/deployment"
        )

        self.api.logs_by_name("test_project", "custom_log")
        mock_make_request.assert_called_with(
            "GET", "by-name/test_project/logs/custom_log"
        )

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_logs_by_uuid(self, mock_make_request):
        self.api.logs_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with("GET", "test_uuid/logs/deployment")

        self.api.logs_by_uuid("test_uuid", "custom_log")
        mock_make_request.assert_called_with("GET", "test_uuid/logs/custom_log")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_delete_by_name(self, mock_make_request):
        self.api.delete_by_name("test_project")
        mock_make_request.assert_called_once_with("DELETE", "by-name/test_project")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_delete_by_uuid(self, mock_make_request):
        self.api.delete_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with("DELETE", "test_uuid")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_list_crews(self, mock_make_request):
        self.api.list_crews()
        mock_make_request.assert_called_once_with("GET", "")

    @patch("crewai.cli.deploy.api.CrewAPI._make_request")
    def test_create_crew(self, mock_make_request):
        payload = {"name": "test_crew"}
        self.api.create_crew(payload)
        mock_make_request.assert_called_once_with("POST", "", json=payload)

    @patch.dict(environ, {"CREWAI_BASE_URL": "https://custom-url.com/api"})
    def test_custom_base_url(self):
        custom_api = CrewAPI("test_key")
        self.assertEqual(
            custom_api.base_url,
            "https://custom-url.com/api",
        )

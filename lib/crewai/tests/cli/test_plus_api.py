import os
import unittest
from unittest.mock import ANY, MagicMock, patch

from crewai.cli.plus_api import PlusAPI


class TestPlusAPI(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api = PlusAPI(self.api_key)
        self.org_uuid = "test-org-uuid"

    def test_init(self):
        self.assertEqual(self.api.api_key, self.api_key)
        self.assertEqual(self.api.headers["Authorization"], f"Bearer {self.api_key}")
        self.assertEqual(self.api.headers["Content-Type"], "application/json")
        self.assertTrue("CrewAI-CLI/" in self.api.headers["User-Agent"])
        self.assertTrue(self.api.headers["X-Crewai-Version"])

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_login_to_tool_repository(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.login_to_tool_repository()

        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/tools/login"
        )
        self.assertEqual(response, mock_response)

    def assert_request_with_org_id(
        self, mock_make_request, method: str, endpoint: str, **kwargs
    ):
        mock_make_request.assert_called_once_with(
            method,
            f"{os.getenv('CREWAI_PLUS_URL')}{endpoint}",
            headers={
                "Authorization": ANY,
                "Content-Type": ANY,
                "User-Agent": ANY,
                "X-Crewai-Version": ANY,
                "X-Crewai-Organization-Id": self.org_uuid,
            },
            **kwargs,
        )

    @patch("crewai.cli.plus_api.Settings")
    @patch("requests.Session.request")
    def test_login_to_tool_repository_with_org_uuid(
        self, mock_make_request, mock_settings_class
    ):
        mock_settings = MagicMock()
        mock_settings.org_uuid = self.org_uuid
        mock_settings.enterprise_base_url = os.getenv('CREWAI_PLUS_URL')
        mock_settings_class.return_value = mock_settings
        # re-initialize Client
        self.api = PlusAPI(self.api_key)

        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.login_to_tool_repository()

        self.assert_request_with_org_id(
            mock_make_request, "POST", "/crewai_plus/api/v1/tools/login"
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_get_agent(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.get_agent("test_agent_handle")
        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/agents/test_agent_handle"
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.Settings")
    @patch("requests.Session.request")
    def test_get_agent_with_org_uuid(self, mock_make_request, mock_settings_class):
        mock_settings = MagicMock()
        mock_settings.org_uuid = self.org_uuid
        mock_settings.enterprise_base_url = os.getenv('CREWAI_PLUS_URL')
        mock_settings_class.return_value = mock_settings
        # re-initialize Client
        self.api = PlusAPI(self.api_key)

        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.get_agent("test_agent_handle")

        self.assert_request_with_org_id(
            mock_make_request, "GET", "/crewai_plus/api/v1/agents/test_agent_handle"
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_get_tool(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.get_tool("test_tool_handle")
        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/tools/test_tool_handle"
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.Settings")
    @patch("requests.Session.request")
    def test_get_tool_with_org_uuid(self, mock_make_request, mock_settings_class):
        mock_settings = MagicMock()
        mock_settings.org_uuid = self.org_uuid
        mock_settings.enterprise_base_url = os.getenv('CREWAI_PLUS_URL')
        mock_settings_class.return_value = mock_settings
        # re-initialize Client
        self.api = PlusAPI(self.api_key)

        # Set up mock response
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.get_tool("test_tool_handle")

        self.assert_request_with_org_id(
            mock_make_request, "GET", "/crewai_plus/api/v1/tools/test_tool_handle"
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_publish_tool(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response
        handle = "test_tool_handle"
        public = True
        version = "1.0.0"
        description = "Test tool description"
        encoded_file = "encoded_test_file"

        response = self.api.publish_tool(
            handle, public, version, description, encoded_file
        )

        params = {
            "handle": handle,
            "public": public,
            "version": version,
            "file": encoded_file,
            "description": description,
            "available_exports": None,
        }
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/tools", json=params
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.Settings")
    @patch("requests.Session.request")
    def test_publish_tool_with_org_uuid(self, mock_make_request, mock_settings_class):
        mock_settings = MagicMock()
        mock_settings.org_uuid = self.org_uuid
        mock_settings.enterprise_base_url = os.getenv('CREWAI_PLUS_URL')
        mock_settings_class.return_value = mock_settings
        # re-initialize Client
        self.api = PlusAPI(self.api_key)

        # Set up mock response
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        handle = "test_tool_handle"
        public = True
        version = "1.0.0"
        description = "Test tool description"
        encoded_file = "encoded_test_file"

        response = self.api.publish_tool(
            handle, public, version, description, encoded_file
        )

        # Expected params including organization_uuid
        expected_params = {
            "handle": handle,
            "public": public,
            "version": version,
            "file": encoded_file,
            "description": description,
            "available_exports": None,
        }

        self.assert_request_with_org_id(
            mock_make_request, "POST", "/crewai_plus/api/v1/tools", json=expected_params
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_publish_tool_without_description(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response
        handle = "test_tool_handle"
        public = False
        version = "2.0.0"
        description = None
        encoded_file = "encoded_test_file"

        response = self.api.publish_tool(
            handle, public, version, description, encoded_file
        )

        params = {
            "handle": handle,
            "public": public,
            "version": version,
            "file": encoded_file,
            "description": description,
            "available_exports": None,
        }
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/tools", json=params
        )
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.requests.Session")
    def test_make_request(self, mock_session):
        mock_response = MagicMock()

        mock_session_instance = mock_session.return_value
        mock_session_instance.request.return_value = mock_response

        response = self.api._make_request("GET", "test_endpoint")

        mock_session.assert_called_once()
        mock_session_instance.request.assert_called_once_with(
            "GET", f"{self.api.base_url}/test_endpoint", headers=self.api.headers
        )
        mock_session_instance.trust_env = False
        self.assertEqual(response, mock_response)

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_deploy_by_name(self, mock_make_request):
        self.api.deploy_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/crews/by-name/test_project/deploy"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_deploy_by_uuid(self, mock_make_request):
        self.api.deploy_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/crews/test_uuid/deploy"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_crew_status_by_name(self, mock_make_request):
        self.api.crew_status_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/crews/by-name/test_project/status"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_crew_status_by_uuid(self, mock_make_request):
        self.api.crew_status_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/crews/test_uuid/status"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_crew_by_name(self, mock_make_request):
        self.api.crew_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/crews/by-name/test_project/logs/deployment"
        )

        self.api.crew_by_name("test_project", "custom_log")
        mock_make_request.assert_called_with(
            "GET", "/crewai_plus/api/v1/crews/by-name/test_project/logs/custom_log"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_crew_by_uuid(self, mock_make_request):
        self.api.crew_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/crews/test_uuid/logs/deployment"
        )

        self.api.crew_by_uuid("test_uuid", "custom_log")
        mock_make_request.assert_called_with(
            "GET", "/crewai_plus/api/v1/crews/test_uuid/logs/custom_log"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_delete_crew_by_name(self, mock_make_request):
        self.api.delete_crew_by_name("test_project")
        mock_make_request.assert_called_once_with(
            "DELETE", "/crewai_plus/api/v1/crews/by-name/test_project"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_delete_crew_by_uuid(self, mock_make_request):
        self.api.delete_crew_by_uuid("test_uuid")
        mock_make_request.assert_called_once_with(
            "DELETE", "/crewai_plus/api/v1/crews/test_uuid"
        )

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_list_crews(self, mock_make_request):
        self.api.list_crews()
        mock_make_request.assert_called_once_with("GET", "/crewai_plus/api/v1/crews")

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_create_crew(self, mock_make_request):
        payload = {"name": "test_crew"}
        self.api.create_crew(payload)
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/crews", json=payload
        )

    @patch("crewai.cli.plus_api.Settings")
    @patch.dict(os.environ, {"CREWAI_PLUS_URL": ""})
    def test_custom_base_url(self, mock_settings_class):
        mock_settings = MagicMock()
        mock_settings.enterprise_base_url = "https://custom-url.com/api"
        mock_settings_class.return_value = mock_settings
        custom_api = PlusAPI("test_key")
        self.assertEqual(
            custom_api.base_url,
            "https://custom-url.com/api",
        )

    @patch.dict(os.environ, {"CREWAI_PLUS_URL": "https://custom-url-from-env.com"})
    def test_custom_base_url_from_env(self):
        custom_api = PlusAPI("test_key")
        self.assertEqual(
            custom_api.base_url,
            "https://custom-url-from-env.com",
        )

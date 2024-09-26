import unittest
from unittest.mock import MagicMock, patch
from crewai.cli.plus_api import PlusAPI


class TestPlusAPI(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.api = PlusAPI(self.api_key)

    @patch("crewai.cli.plus_api.PlusAPI._make_request")
    def test_get_tool(self, mock_make_request):
        mock_response = MagicMock()
        mock_make_request.return_value = mock_response

        response = self.api.get_tool("test_tool_handle")

        mock_make_request.assert_called_once_with(
            "GET", "/crewai_plus/api/v1/tools/test_tool_handle"
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
        }
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/tools", json=params
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
        }
        mock_make_request.assert_called_once_with(
            "POST", "/crewai_plus/api/v1/tools", json=params
        )
        self.assertEqual(response, mock_response)

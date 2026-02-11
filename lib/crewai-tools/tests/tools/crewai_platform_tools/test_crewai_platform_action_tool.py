from unittest.mock import patch, Mock
import os

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)


class TestCrewAIPlatformActionToolVerify:
    """Test suite for SSL verification behavior based on CREWAI_FACTORY environment variable"""

    def setup_method(self):
        self.action_schema = {
            "function": {
                "name": "test_action",
                "parameters": {
                    "properties": {
                        "test_param": {
                            "type": "string",
                            "description": "Test parameter"
                        }
                    },
                    "required": []
                }
            }
        }

    def create_test_tool(self):
        return CrewAIPlatformActionTool(
            description="Test action tool",
            action_name="test_action",
            action_schema=self.action_schema
        )

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post")
    def test_run_with_ssl_verification_default(self, mock_post):
        """Test that _run uses SSL verification by default when CREWAI_FACTORY is not set"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "false"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post")
    def test_run_with_ssl_verification_factory_false(self, mock_post):
        """Test that _run uses SSL verification when CREWAI_FACTORY is 'false'"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "FALSE"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post")
    def test_run_with_ssl_verification_factory_false_uppercase(self, mock_post):
        """Test that _run uses SSL verification when CREWAI_FACTORY is 'FALSE' (case-insensitive)"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "true"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post")
    def test_run_without_ssl_verification_factory_true(self, mock_post):
        """Test that _run disables SSL verification when CREWAI_FACTORY is 'true'"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is False

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "TRUE"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post")
    def test_run_without_ssl_verification_factory_true_uppercase(self, mock_post):
        """Test that _run disables SSL verification when CREWAI_FACTORY is 'TRUE' (case-insensitive)"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is False

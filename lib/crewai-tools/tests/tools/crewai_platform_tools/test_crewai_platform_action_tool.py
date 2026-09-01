import os
from typing import cast
from unittest.mock import Mock, patch

from crewai.tools.tool_failure import ToolFailure

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)
from crewai_tools.tools.crewai_platform_tools.integrations_client import (
    IntegrationsClient,
    ToolExecutionFailure,
    ToolExecutionSuccess,
    ToolInfo,
)


class TestCrewAIPlatformActionToolVerify:
    """Test suite for SSL verification behavior based on CREWAI_FACTORY environment variable"""

    def setup_method(self):
        self.tool_info = ToolInfo(
            app="test_app",
            action="test_action",
            connection_id=None,
            description="Test action tool",
            parameters={
                "properties": {
                    "test_param": {
                        "type": "string",
                        "description": "Test parameter",
                    }
                },
                "required": [],
            },
        )

    def create_test_tool(
        self, client: IntegrationsClient | None = None
    ) -> CrewAIPlatformActionTool:
        return CrewAIPlatformActionTool(self.tool_info, client=client)

    def test_run_serializes_success_output(self):
        client = Mock(spec=IntegrationsClient)
        client.execute_action.return_value = ToolExecutionSuccess(
            output={"result": {"id": 42}}
        )

        result = self.create_test_tool(cast(IntegrationsClient, client))._run(
            test_param="test_value", optional_param=None
        )

        assert result == '{\n  "result": {\n    "id": 42\n  }\n}'
        assert client.execute_action.call_args.args[1] == {"test_param": "test_value"}

    def test_run_converts_expected_failure(self):
        client = Mock(spec=IntegrationsClient)
        client.execute_action.return_value = ToolExecutionFailure(
            message="Channel not found",
            code="404",
            retryable=False,
        )

        result = self.create_test_tool(cast(IntegrationsClient, client))._run(
            test_param="test_value"
        )

        assert result == ToolFailure(
            message="API request failed: Channel not found",
            code="404",
            retryable=False,
            details={"action": "test_action"},
        )

    def test_run_preserves_unexpected_exception_fallback(self):
        client = Mock(spec=IntegrationsClient)
        client.execute_action.side_effect = ValueError("Invalid response JSON")

        result = self.create_test_tool(cast(IntegrationsClient, client))._run(
            test_param="test_value"
        )

        assert result == ToolFailure(
            message="Error executing action test_action: Invalid response JSON",
            code="ValueError",
            details={"action": "test_action"},
        )

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post")
    def test_run_with_ssl_verification_default(self, mock_post):
        """Test that _run uses SSL verification by default when CREWAI_FACTORY is not set"""
        mock_response = Mock(status_code=200)
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "false"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post")
    def test_run_with_ssl_verification_factory_false(self, mock_post):
        """Test that _run uses SSL verification when CREWAI_FACTORY is 'false'"""
        mock_response = Mock(status_code=200)
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "FALSE"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post")
    def test_run_with_ssl_verification_factory_false_uppercase(self, mock_post):
        """Test that _run uses SSL verification when CREWAI_FACTORY is 'FALSE' (case-insensitive)"""
        mock_response = Mock(status_code=200)
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "true"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post")
    def test_run_without_ssl_verification_factory_true(self, mock_post):
        """Test that _run disables SSL verification when CREWAI_FACTORY is 'true'"""
        mock_response = Mock(status_code=200)
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is False

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "TRUE"}, clear=True)
    @patch("crewai_tools.tools.crewai_platform_tools.integrations_client.requests.post")
    def test_run_without_ssl_verification_factory_true_uppercase(self, mock_post):
        """Test that _run disables SSL verification when CREWAI_FACTORY is 'TRUE' (case-insensitive)"""
        mock_response = Mock(status_code=200)
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        tool = self.create_test_tool()
        tool._run(test_param="test_value")

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is False

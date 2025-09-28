import unittest
from unittest.mock import Mock, patch

from crewai_tools.tools.crewai_platform_tools import CrewAIPlatformActionTool
import pytest


class TestCrewAIPlatformActionTool(unittest.TestCase):
    @pytest.fixture
    def sample_action_schema(self):
        return {
            "function": {
                "name": "test_action",
                "description": "Test action for unit testing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to send"},
                        "priority": {
                            "type": "integer",
                            "description": "Priority level",
                        },
                    },
                    "required": ["message"],
                },
            }
        }

    @pytest.fixture
    def platform_action_tool(self, sample_action_schema):
        return CrewAIPlatformActionTool(
            description="Test Action Tool\nTest description",
            action_name="test_action",
            action_schema=sample_action_schema,
        )

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
    )
    def test_run_success(self, mock_post):
        schema = {
            "function": {
                "name": "test_action",
                "description": "Test action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message"}
                    },
                    "required": ["message"],
                },
            }
        }

        tool = CrewAIPlatformActionTool(
            description="Test tool", action_name="test_action", action_schema=schema
        )

        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"result": "success", "data": "test_data"}
        mock_post.return_value = mock_response

        result = tool._run(message="test message")

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args

        assert "test_action/execute" in kwargs["url"]
        assert kwargs["headers"]["Authorization"] == "Bearer test_token"
        assert kwargs["json"]["message"] == "test message"
        assert "success" in result

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
    )
    def test_run_api_error(self, mock_post):
        schema = {
            "function": {
                "name": "test_action",
                "description": "Test action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message"}
                    },
                    "required": ["message"],
                },
            }
        }

        tool = CrewAIPlatformActionTool(
            description="Test tool", action_name="test_action", action_schema=schema
        )

        mock_response = Mock()
        mock_response.ok = False
        mock_response.json.return_value = {"error": {"message": "Invalid request"}}
        mock_post.return_value = mock_response

        result = tool._run(message="test message")

        assert "API request failed" in result
        assert "Invalid request" in result

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool.requests.post"
    )
    def test_run_exception(self, mock_post):
        schema = {
            "function": {
                "name": "test_action",
                "description": "Test action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message"}
                    },
                    "required": ["message"],
                },
            }
        }

        tool = CrewAIPlatformActionTool(
            description="Test tool", action_name="test_action", action_schema=schema
        )

        mock_post.side_effect = Exception("Network error")

        result = tool._run(message="test message")

        assert "Error executing action test_action: Network error" in result

    def test_run_without_token(self):
        schema = {
            "function": {
                "name": "test_action",
                "description": "Test action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message"}
                    },
                    "required": ["message"],
                },
            }
        }

        tool = CrewAIPlatformActionTool(
            description="Test tool", action_name="test_action", action_schema=schema
        )

        with patch.dict("os.environ", {}, clear=True):
            result = tool._run(message="test message")
            assert "Error executing action test_action:" in result
            assert "No platform integration token found" in result

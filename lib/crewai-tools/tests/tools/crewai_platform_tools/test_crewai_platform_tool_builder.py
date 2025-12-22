import unittest
from unittest.mock import Mock, patch

from crewai_tools.tools.crewai_platform_tools import (
    CrewAIPlatformActionTool,
    CrewaiPlatformToolBuilder,
)
import pytest


class TestCrewaiPlatformToolBuilder(unittest.TestCase):
    @pytest.fixture
    def platform_tool_builder(self):
        """Create a CrewaiPlatformToolBuilder instance for testing"""
        return CrewaiPlatformToolBuilder(apps=["github", "slack"])

    @pytest.fixture
    def mock_api_response(self):
        return {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Issue title",
                                },
                                "body": {"type": "string", "description": "Issue body"},
                            },
                            "required": ["title"],
                        },
                    }
                ],
                "slack": [
                    {
                        "name": "send_message",
                        "description": "Send a Slack message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Channel name",
                                },
                                "text": {
                                    "type": "string",
                                    "description": "Message text",
                                },
                            },
                            "required": ["channel", "text"],
                        },
                    }
                ],
            }
        }

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_fetch_actions_success(self, mock_get):
        mock_api_response = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Issue title",
                                }
                            },
                            "required": ["title"],
                        },
                    }
                ]
            }
        }

        builder = CrewaiPlatformToolBuilder(apps=["github", "slack/send_message"])

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response

        builder._fetch_actions()

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args

        assert "/actions" in args[0]
        assert kwargs["headers"]["Authorization"] == "Bearer test_token"
        assert kwargs["params"]["apps"] == "github,slack/send_message"

        assert "create_issue" in builder._actions_schema
        assert (
            builder._actions_schema["create_issue"]["function"]["name"]
            == "create_issue"
        )

    def test_fetch_actions_no_token(self):
        builder = CrewaiPlatformToolBuilder(apps=["github"])

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError) as context:
                builder._fetch_actions()
            assert "No platform integration token found" in str(context.exception)

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_create_tools(self, mock_get):
        mock_api_response = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create a GitHub issue",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Issue title",
                                }
                            },
                            "required": ["title"],
                        },
                    }
                ],
                "slack": [
                    {
                        "name": "send_message",
                        "description": "Send a Slack message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "channel": {
                                    "type": "string",
                                    "description": "Channel name",
                                }
                            },
                            "required": ["channel"],
                        },
                    }
                ],
            }
        }

        builder = CrewaiPlatformToolBuilder(apps=["github", "slack"])

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response

        tools = builder.tools()

        assert len(tools) == 2
        assert all(isinstance(tool, CrewAIPlatformActionTool) for tool in tools)

        tool_names = [tool.action_name for tool in tools]
        assert "create_issue" in tool_names
        assert "send_message" in tool_names

        github_tool = next((t for t in tools if t.action_name == "create_issue"), None)
        slack_tool = next((t for t in tools if t.action_name == "send_message"), None)

        assert github_tool is not None
        assert slack_tool is not None
        assert "Create a GitHub issue" in github_tool.description
        assert "Send a Slack message" in slack_tool.description

    def test_tools_caching(self):
        builder = CrewaiPlatformToolBuilder(apps=["github"])

        cached_tools = []

        def mock_create_tools():
            builder._tools = cached_tools

        with (
            patch.object(builder, "_fetch_actions") as mock_fetch,
            patch.object(
                builder, "_create_tools", side_effect=mock_create_tools
            ) as mock_create,
        ):
            tools1 = builder.tools()
            assert mock_fetch.call_count == 1
            assert mock_create.call_count == 1

            tools2 = builder.tools()
            assert mock_fetch.call_count == 1
            assert mock_create.call_count == 1

            assert tools1 is tools2

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    def test_empty_apps_list(self):
        builder = CrewaiPlatformToolBuilder(apps=[])

        with patch(
            "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"actions": {}}
            mock_get.return_value = mock_response

            tools = builder.tools()

            assert isinstance(tools, list)
            assert len(tools) == 0

            _, kwargs = mock_get.call_args
            assert kwargs["params"]["apps"] == ""

    def test_detailed_description_generation(self):
        builder = CrewaiPlatformToolBuilder(apps=["test"])

        complex_schema = {
            "type": "object",
            "properties": {
                "simple_string": {"type": "string", "description": "A simple string"},
                "nested_object": {
                    "type": "object",
                    "properties": {
                        "inner_prop": {
                            "type": "integer",
                            "description": "Inner property",
                        }
                    },
                    "description": "Nested object",
                },
                "array_prop": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of strings",
                },
            },
        }

        descriptions = builder._generate_detailed_description(complex_schema)

        assert isinstance(descriptions, list)
        assert len(descriptions) > 0

        description_text = "\n".join(descriptions)
        assert "simple_string" in description_text
        assert "nested_object" in description_text
        assert "array_prop" in description_text



class TestCrewaiPlatformToolBuilderVerify(unittest.TestCase):
    """Test suite for SSL verification behavior in CrewaiPlatformToolBuilder"""

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"}, clear=True)
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_fetch_actions_with_ssl_verification_default(self, mock_get):
        """Test that _fetch_actions uses SSL verification by default when CREWAI_FACTORY is not set"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {}}
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(apps=["github"])
        builder._fetch_actions()

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "false"}, clear=True)
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_fetch_actions_with_ssl_verification_factory_false(self, mock_get):
        """Test that _fetch_actions uses SSL verification when CREWAI_FACTORY is 'false'"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {}}
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(apps=["github"])
        builder._fetch_actions()

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "FALSE"}, clear=True)
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_fetch_actions_with_ssl_verification_factory_false_uppercase(self, mock_get):
        """Test that _fetch_actions uses SSL verification when CREWAI_FACTORY is 'FALSE' (case-insensitive)"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {}}
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(apps=["github"])
        builder._fetch_actions()

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["verify"] is True

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "true"}, clear=True)
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_fetch_actions_without_ssl_verification_factory_true(self, mock_get):
        """Test that _fetch_actions disables SSL verification when CREWAI_FACTORY is 'true'"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {}}
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(apps=["github"])
        builder._fetch_actions()

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["verify"] is False

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token", "CREWAI_FACTORY": "TRUE"}, clear=True)
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_fetch_actions_without_ssl_verification_factory_true_uppercase(self, mock_get):
        """Test that _fetch_actions disables SSL verification when CREWAI_FACTORY is 'TRUE' (case-insensitive)"""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"actions": {}}
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(apps=["github"])
        builder._fetch_actions()

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs["verify"] is False

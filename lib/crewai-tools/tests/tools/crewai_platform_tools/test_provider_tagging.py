"""Tests for the _provider / _provider_id metadata attached to platform tools.

These attributes are private metadata used by enterprise tooling to recover
the canonical tool_id (e.g. ``crewai_oauth:google_drive`` or
``paragon:<uuid>``) for ACP rule evaluation.
"""

from unittest.mock import Mock, patch

from crewai_tools.tools.crewai_platform_tools import (
    CrewAIPlatformActionTool,
    CrewaiPlatformToolBuilder,
)


class TestActionToolProviderAttrs:
    def setup_method(self):
        self.action_schema = {
            "function": {
                "name": "test_action",
                "parameters": {"type": "object", "properties": {}},
            }
        }

    def test_defaults_to_none_when_not_provided(self):
        tool = CrewAIPlatformActionTool(
            description="x",
            action_name="test_action",
            action_schema=self.action_schema,
        )
        assert tool._provider is None
        assert tool._provider_id is None

    def test_stores_explicit_values(self):
        tool = CrewAIPlatformActionTool(
            description="x",
            action_name="test_action",
            action_schema=self.action_schema,
            provider="crewai_oauth",
            provider_id="google_drive",
        )
        assert tool._provider == "crewai_oauth"
        assert tool._provider_id == "google_drive"


class TestBuilderProviderThreading:
    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_builder_threads_provider_and_app_into_each_tool(self, mock_get):
        mock_api_response = {
            "actions": {
                "google_drive": [
                    {
                        "name": "create_file",
                        "description": "Create a file",
                        "parameters": {"type": "object", "properties": {}},
                        "provider": "crewai_oauth",
                    }
                ],
                "1b5f2395-65a5-4da8-9b2f-c10eafc83a0b": [
                    {
                        "name": "send_invoice",
                        "description": "Send an invoice",
                        "parameters": {"type": "object", "properties": {}},
                        "provider": "paragon",
                    }
                ],
            }
        }
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(
            apps=["google_drive", "1b5f2395-65a5-4da8-9b2f-c10eafc83a0b"]
        )
        tools = builder.tools()

        by_action = {tool.action_name: tool for tool in tools}

        oauth_tool = by_action["create_file"]
        assert oauth_tool._provider == "crewai_oauth"
        assert oauth_tool._provider_id == "google_drive"

        paragon_tool = by_action["send_invoice"]
        assert paragon_tool._provider == "paragon"
        assert paragon_tool._provider_id == "1b5f2395-65a5-4da8-9b2f-c10eafc83a0b"

    @patch.dict("os.environ", {"CREWAI_PLATFORM_INTEGRATION_TOKEN": "test_token"})
    @patch(
        "crewai_tools.tools.crewai_platform_tools.crewai_platform_tool_builder.requests.get"
    )
    def test_builder_handles_response_without_provider_field(self, mock_get):
        # Older crewai-plus versions return actions without a "provider" key.
        # The builder must remain compatible: provider_id is set, provider is None.
        mock_api_response = {
            "actions": {
                "github": [
                    {
                        "name": "create_issue",
                        "description": "Create issue",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            }
        }
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = mock_api_response
        mock_get.return_value = mock_response

        builder = CrewaiPlatformToolBuilder(apps=["github"])
        tools = builder.tools()

        assert len(tools) == 1
        assert tools[0]._provider is None
        assert tools[0]._provider_id == "github"

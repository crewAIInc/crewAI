import os
import unittest
from unittest.mock import patch, MagicMock

from crewai.tools import BaseTool
from crewai_tools.tools import CrewaiEnterpriseTools
from crewai_tools.adapters.tool_collection import ToolCollection


class TestCrewaiEnterpriseTools(unittest.TestCase):
    def setUp(self):
        self.mock_tools = [
            self._create_mock_tool("tool1", "Tool 1 Description"),
            self._create_mock_tool("tool2", "Tool 2 Description"),
            self._create_mock_tool("tool3", "Tool 3 Description"),
        ]
        self.adapter_patcher = patch(
            "crewai_tools.tools.crewai_enterprise_tools.crewai_enterprise_tools.EnterpriseActionKitToolAdapter"
        )
        self.MockAdapter = self.adapter_patcher.start()

        mock_adapter_instance = self.MockAdapter.return_value
        mock_adapter_instance.tools.return_value = self.mock_tools

    def tearDown(self):
        self.adapter_patcher.stop()

    def _create_mock_tool(self, name, description):
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = name
        mock_tool.description = description
        return mock_tool

    @patch.dict(os.environ, {"CREWAI_ENTERPRISE_TOOLS_TOKEN": "env-token"})
    def test_returns_tool_collection(self):
        tools = CrewaiEnterpriseTools()
        self.assertIsInstance(tools, ToolCollection)

    @patch.dict(os.environ, {"CREWAI_ENTERPRISE_TOOLS_TOKEN": "env-token"})
    def test_returns_all_tools_when_no_actions_list(self):
        tools = CrewaiEnterpriseTools()
        self.assertEqual(len(tools), 3)
        self.assertEqual(tools[0].name, "tool1")
        self.assertEqual(tools[1].name, "tool2")
        self.assertEqual(tools[2].name, "tool3")

    @patch.dict(os.environ, {"CREWAI_ENTERPRISE_TOOLS_TOKEN": "env-token"})
    def test_filters_tools_by_actions_list(self):
        tools = CrewaiEnterpriseTools(actions_list=["ToOl1", "tool3"])
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0].name, "tool1")
        self.assertEqual(tools[1].name, "tool3")

    def test_uses_provided_parameters(self):
        CrewaiEnterpriseTools(
            enterprise_token="test-token",
            enterprise_action_kit_project_id="project-id",
            enterprise_action_kit_project_url="project-url",
        )

        self.MockAdapter.assert_called_once_with(
            enterprise_action_token="test-token",
            enterprise_action_kit_project_id="project-id",
            enterprise_action_kit_project_url="project-url",
        )

    @patch.dict(os.environ, {"CREWAI_ENTERPRISE_TOOLS_TOKEN": "env-token"})
    def test_uses_environment_token(self):
        CrewaiEnterpriseTools()
        self.MockAdapter.assert_called_once_with(enterprise_action_token="env-token")

    @patch.dict(os.environ, {"CREWAI_ENTERPRISE_TOOLS_TOKEN": "env-token"})
    def test_uses_environment_token_when_no_token_provided(self):
        CrewaiEnterpriseTools(enterprise_token="")
        self.MockAdapter.assert_called_once_with(enterprise_action_token="env-token")

    @patch.dict(
        os.environ,
        {
            "CREWAI_ENTERPRISE_TOOLS_TOKEN": "env-token",
            "CREWAI_ENTERPRISE_TOOLS_ACTIONS_LIST": '["tool1", "tool3"]',
        },
    )
    def test_uses_environment_actions_list(self):
        tools = CrewaiEnterpriseTools()
        self.assertEqual(len(tools), 2)
        self.assertEqual(tools[0].name, "tool1")
        self.assertEqual(tools[1].name, "tool3")

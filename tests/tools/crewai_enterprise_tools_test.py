import os
import unittest
from unittest.mock import patch, MagicMock


from crewai.tools import BaseTool
from crewai_tools.tools import CrewaiEnterpriseTools
from crewai_tools.adapters.tool_collection import ToolCollection
from crewai_tools.adapters.enterprise_adapter import EnterpriseActionTool


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


class TestEnterpriseActionToolSchemaConversion(unittest.TestCase):
    """Test the enterprise action tool schema conversion and validation."""

    def setUp(self):
        self.test_schema = {
            "type": "function",
            "function": {
                "name": "TEST_COMPLEX_ACTION",
                "description": "Test action with complex nested structure",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filterCriteria": {
                            "type": "object",
                            "description": "Filter criteria object",
                            "properties": {
                                "operation": {"type": "string", "enum": ["AND", "OR"]},
                                "rules": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "field": {
                                                "type": "string",
                                                "enum": ["name", "email", "status"],
                                            },
                                            "operator": {
                                                "type": "string",
                                                "enum": ["equals", "contains"],
                                            },
                                            "value": {"type": "string"},
                                        },
                                        "required": ["field", "operator", "value"],
                                    },
                                },
                            },
                            "required": ["operation", "rules"],
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "limit": {"type": "integer"},
                                "offset": {"type": "integer"},
                            },
                            "required": [],
                        },
                    },
                    "required": [],
                },
            },
        }

    def test_complex_schema_conversion(self):
        """Test that complex nested schemas are properly converted to Pydantic models."""
        tool = EnterpriseActionTool(
            name="gmail_search_for_email",
            description="Test tool",
            enterprise_action_token="test_token",
            action_name="GMAIL_SEARCH_FOR_EMAIL",
            action_schema=self.test_schema,
        )

        self.assertEqual(tool.name, "gmail_search_for_email")
        self.assertEqual(tool.action_name, "GMAIL_SEARCH_FOR_EMAIL")

        schema_class = tool.args_schema
        self.assertIsNotNone(schema_class)

        schema_fields = schema_class.model_fields
        self.assertIn("filterCriteria", schema_fields)
        self.assertIn("options", schema_fields)

        # Test valid input structure
        valid_input = {
            "filterCriteria": {
                "operation": "AND",
                "rules": [
                    {"field": "name", "operator": "contains", "value": "test"},
                    {"field": "status", "operator": "equals", "value": "active"},
                ],
            },
            "options": {"limit": 10},
        }

        # This should not raise an exception
        validated_input = schema_class(**valid_input)
        self.assertIsNotNone(validated_input.filterCriteria)
        self.assertIsNotNone(validated_input.options)

    def test_optional_fields_validation(self):
        """Test that optional fields work correctly."""
        tool = EnterpriseActionTool(
            name="gmail_search_for_email",
            description="Test tool",
            enterprise_action_token="test_token",
            action_name="GMAIL_SEARCH_FOR_EMAIL",
            action_schema=self.test_schema,
        )

        schema_class = tool.args_schema

        minimal_input = {}
        validated_input = schema_class(**minimal_input)
        self.assertIsNone(validated_input.filterCriteria)
        self.assertIsNone(validated_input.options)

        partial_input = {"options": {"limit": 10}}
        validated_input = schema_class(**partial_input)
        self.assertIsNone(validated_input.filterCriteria)
        self.assertIsNotNone(validated_input.options)

    def test_enum_validation(self):
        """Test that enum values are properly validated."""
        tool = EnterpriseActionTool(
            name="gmail_search_for_email",
            description="Test tool",
            enterprise_action_token="test_token",
            action_name="GMAIL_SEARCH_FOR_EMAIL",
            action_schema=self.test_schema,
        )

        schema_class = tool.args_schema

        invalid_input = {
            "filterCriteria": {
                "operation": "INVALID_OPERATOR",
                "rules": [],
            }
        }

        with self.assertRaises(Exception):
            schema_class(**invalid_input)

    def test_required_nested_fields(self):
        """Test that required fields in nested objects are validated."""
        tool = EnterpriseActionTool(
            name="gmail_search_for_email",
            description="Test tool",
            enterprise_action_token="test_token",
            action_name="GMAIL_SEARCH_FOR_EMAIL",
            action_schema=self.test_schema,
        )

        schema_class = tool.args_schema

        incomplete_input = {
            "filterCriteria": {
                "operation": "OR",
                "rules": [
                    {
                        "field": "name",
                        "operator": "contains",
                    }
                ],
            }
        }

        with self.assertRaises(Exception):
            schema_class(**incomplete_input)

    @patch("requests.post")
    def test_tool_execution_with_complex_input(self, mock_post):
        """Test that the tool can execute with complex validated input."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"success": True, "results": []}
        mock_post.return_value = mock_response

        tool = EnterpriseActionTool(
            name="gmail_search_for_email",
            description="Test tool",
            enterprise_action_token="test_token",
            action_name="GMAIL_SEARCH_FOR_EMAIL",
            action_schema=self.test_schema,
        )

        tool._run(
            filterCriteria={
                "operation": "OR",
                "rules": [
                    {"field": "name", "operator": "contains", "value": "test"},
                    {"field": "status", "operator": "equals", "value": "active"},
                ],
            },
            options={"limit": 10},
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]["json"]

        self.assertEqual(payload["action"], "GMAIL_SEARCH_FOR_EMAIL")
        self.assertIn("filterCriteria", payload["parameters"])
        self.assertIn("options", payload["parameters"])
        self.assertEqual(payload["parameters"]["filterCriteria"]["operation"], "OR")

    def test_model_naming_convention(self):
        """Test that generated model names follow proper conventions."""
        tool = EnterpriseActionTool(
            name="gmail_search_for_email",
            description="Test tool",
            enterprise_action_token="test_token",
            action_name="GMAIL_SEARCH_FOR_EMAIL",
            action_schema=self.test_schema,
        )

        schema_class = tool.args_schema
        self.assertIsNotNone(schema_class)

        self.assertTrue(schema_class.__name__.endswith("Schema"))
        self.assertTrue(schema_class.__name__[0].isupper())

        complex_input = {
            "filterCriteria": {
                "operation": "OR",
                "rules": [
                    {"field": "name", "operator": "contains", "value": "test"},
                    {"field": "status", "operator": "equals", "value": "active"},
                ],
            },
            "options": {"limit": 10},
        }

        validated = schema_class(**complex_input)
        self.assertIsNotNone(validated.filterCriteria)

    def test_simple_schema_with_enums(self):
        """Test a simpler schema with basic enum validation."""
        simple_schema = {
            "type": "function",
            "function": {
                "name": "SIMPLE_TEST",
                "description": "Simple test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive", "pending"],
                        },
                        "priority": {"type": "integer", "enum": [1, 2, 3]},
                    },
                    "required": ["status"],
                },
            },
        }

        tool = EnterpriseActionTool(
            name="simple_test",
            description="Simple test tool",
            enterprise_action_token="test_token",
            action_name="SIMPLE_TEST",
            action_schema=simple_schema,
        )

        schema_class = tool.args_schema

        valid_input = {"status": "active", "priority": 2}
        validated = schema_class(**valid_input)
        self.assertEqual(validated.status, "active")
        self.assertEqual(validated.priority, 2)

        with self.assertRaises(Exception):
            schema_class(status="invalid_status")

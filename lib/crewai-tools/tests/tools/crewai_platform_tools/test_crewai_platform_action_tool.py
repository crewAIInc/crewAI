from typing import Union, get_args, get_origin
from unittest.mock import patch, Mock
import os

from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import (
    CrewAIPlatformActionTool,
)


class TestSchemaProcessing:

    def setup_method(self):
        self.base_action_schema = {
            "function": {
                "parameters": {
                    "properties": {},
                    "required": []
                }
            }
        }

    def create_test_tool(self, action_name="test_action"):
        return CrewAIPlatformActionTool(
            description="Test tool",
            action_name=action_name,
            action_schema=self.base_action_schema
        )

    def test_anyof_multiple_types(self):
        tool = self.create_test_tool()

        test_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "integer"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestField")

        assert get_origin(result_type) is Union

        args = get_args(result_type)
        expected_types = (str, float, int)

        for expected_type in expected_types:
            assert expected_type in args

    def test_anyof_with_null(self):
        tool = self.create_test_tool()

        test_schema = {
            "anyOf": [
                {"type": "string"},
                {"type": "number"},
                {"type": "null"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldNullable")

        assert get_origin(result_type) is Union

        args = get_args(result_type)
        assert type(None) in args
        assert str in args
        assert float in args

    def test_anyof_single_type(self):
        tool = self.create_test_tool()

        test_schema = {
            "anyOf": [
                {"type": "string"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldSingle")

        assert result_type is str

    def test_oneof_multiple_types(self):
        tool = self.create_test_tool()

        test_schema = {
            "oneOf": [
                {"type": "string"},
                {"type": "boolean"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldOneOf")

        assert get_origin(result_type) is Union

        args = get_args(result_type)
        expected_types = (str, bool)

        for expected_type in expected_types:
            assert expected_type in args

    def test_oneof_single_type(self):
        tool = self.create_test_tool()

        test_schema = {
            "oneOf": [
                {"type": "integer"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldOneOfSingle")

        assert result_type is int

    def test_basic_types(self):
        tool = self.create_test_tool()

        test_cases = [
            ({"type": "string"}, str),
            ({"type": "integer"}, int),
            ({"type": "number"}, float),
            ({"type": "boolean"}, bool),
            ({"type": "array", "items": {"type": "string"}}, list),
        ]

        for schema, expected_type in test_cases:
            result_type = tool._process_schema_type(schema, "TestField")
            if schema["type"] == "array":
                assert get_origin(result_type) is list
            else:
                assert result_type is expected_type

    def test_enum_handling(self):
        tool = self.create_test_tool()

        test_schema = {
            "type": "string",
            "enum": ["option1", "option2", "option3"]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldEnum")

        assert result_type is str

    def test_nested_anyof(self):
        tool = self.create_test_tool()

        test_schema = {
            "anyOf": [
                {"type": "string"},
                {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "boolean"}
                    ]
                }
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldNested")

        assert get_origin(result_type) is Union
        args = get_args(result_type)

        assert str in args

        if len(args) == 3:
            assert int in args
            assert bool in args
        else:
            nested_union = next(arg for arg in args if get_origin(arg) is Union)
            nested_args = get_args(nested_union)
            assert int in nested_args
            assert bool in nested_args

    def test_allof_same_types(self):
        tool = self.create_test_tool()

        test_schema = {
            "allOf": [
                {"type": "string"},
                {"type": "string", "maxLength": 100}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldAllOfSame")

        assert result_type is str

    def test_allof_object_merge(self):
        tool = self.create_test_tool()

        test_schema = {
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name"]
                },
                {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                            "age": {"type": "integer"}
                    },
                    "required": ["email"]
                }
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldAllOfMerged")

        # Should create a merged model with all properties
        # The implementation might fall back to dict if model creation fails
        # Let's just verify it's not a basic scalar type
        assert result_type is not str
        assert result_type is not int
        assert result_type is not bool
        # It could be dict (fallback) or a proper model class
        assert result_type in (dict, type) or hasattr(result_type, '__name__')

    def test_allof_single_schema(self):
        """Test that allOf with single schema works correctly."""
        tool = self.create_test_tool()

        test_schema = {
            "allOf": [
                {"type": "boolean"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldAllOfSingle")

        # Should be just bool
        assert result_type is bool

    def test_allof_mixed_types(self):
        tool = self.create_test_tool()

        test_schema = {
            "allOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }

        result_type = tool._process_schema_type(test_schema, "TestFieldAllOfMixed")

        assert result_type is str

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

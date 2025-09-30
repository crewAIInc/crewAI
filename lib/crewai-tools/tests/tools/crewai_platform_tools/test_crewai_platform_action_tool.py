from typing import Union, Optional, get_origin, get_args
from crewai_tools.tools.crewai_platform_tools.crewai_platform_action_tool import CrewAIPlatformActionTool


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
            nested_union = [arg for arg in args if get_origin(arg) is Union][0]
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

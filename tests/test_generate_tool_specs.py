import json
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field
from unittest import mock

from generate_tool_specs import ToolSpecExtractor
from crewai.tools.base_tool import EnvVar

class MockToolSchema(BaseModel):
    query: str = Field(..., description="The query parameter")
    count: int = Field(5, description="Number of results to return")
    filters: Optional[List[str]] = Field(None, description="Optional filters to apply")


class MockTool:
    name = "Mock Search Tool"
    description = "A tool that mocks search functionality"
    args_schema = MockToolSchema

@pytest.fixture
def extractor():
    ext = ToolSpecExtractor()
    MockTool.__pydantic_core_schema__ = create_mock_schema(MockTool)
    MockTool.args_schema.__pydantic_core_schema__ = create_mock_schema_args(MockTool.args_schema)
    return ext


def create_mock_schema(cls):
    return {
        "type": "model",
        "cls": cls,
        "schema": {
            "type": "model-fields",
            "fields": {
                "name": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "str"}, "default": cls.name}, "metadata": {}},
                "description": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "str"}, "default": cls.description}, "metadata": {}},
                "args_schema": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "is-subclass", "cls": BaseModel}, "default": cls.args_schema}, "metadata": {}},
                "env_vars": {
                    "type": "model-field", "schema": {"type": "default", "schema": {"type": "list", "items_schema": {"type": "model", "cls": "INSPECT CLASS", "schema": {"type": "model-fields", "fields": {"name": {"type": "model-field", "schema": {"type": "str"}, "metadata": {}}, "description": {"type": "model-field", "schema": {"type": "str"}, "metadata": {}}, "required": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "bool"}, "default": True}, "metadata": {}}, "default": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "nullable", "schema": {"type": "str"}}, "default": None}, "metadata": {}},}, "model_name": "EnvVar", "computed_fields": []}, "custom_init": False, "root_model": False, "config": {"title": "EnvVar"}, "ref": "crewai.tools.base_tool.EnvVar:4593650640", "metadata": {"pydantic_js_functions": ["INSPECT __get_pydantic_json_schema__"]}}}, "default": [EnvVar(name='SERPER_API_KEY', description='API key for Serper', required=True, default=None), EnvVar(name='API_RATE_LIMIT', description='API rate limit', required=False, default="100")]}, "metadata": {}
                }
            },
            "model_name": cls.__name__
        }
    }


def create_mock_schema_args(cls):
    return {
        "type": "model",
        "cls": cls,
        "schema": {
            "type": "model-fields",
            "fields": {
                "query": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "str"}, "default": "The query parameter"}},
                "count": {"type": "model-field", "schema": {"type": "default", "schema": {"type": "int"}, "default": 5}, "metadata": {"pydantic_js_updates": {"description": "Number of results to return"}}},
                "filters": {"type": "model-field", "schema": {"type": "nullable", "schema": {"type": "list", "items_schema": {"type": "str"}}}}
            },
            "model_name": cls.__name__
        }
    }


def test_unwrap_schema(extractor):
    nested_schema = {
        "type": "function-after",
        "schema": {"type": "default", "schema": {"type": "str", "value": "test"}}
    }
    result = extractor._unwrap_schema(nested_schema)
    assert result["type"] == "str"
    assert result["value"] == "test"


@pytest.mark.parametrize(
    "field, fallback, expected",
    [
        ({"schema": {"default": "test_value"}}, None, "test_value"),
        ({}, "fallback_value", "fallback_value"),
        ({"schema": {"default": 123}}, "fallback_value", "fallback_value")
    ]
)
def test_extract_field_default(extractor, field, fallback, expected):
    result = extractor._extract_field_default(field, fallback=fallback)
    assert result == expected


@pytest.mark.parametrize(
    "schema, expected",
    [
        ({"type": "str"}, "str"),
        ({"type": "list", "items_schema": {"type": "str"}}, "list[str]"),
        ({"type": "dict", "keys_schema": {"type": "str"}, "values_schema": {"type": "int"}}, "dict[str, int]"),
        ({"type": "union", "choices": [{"type": "str"}, {"type": "int"}]}, "union[str, int]"),
        ({"type": "custom_type"}, "custom_type"),
        ({}, "unknown"),
    ]
)
def test_schema_type_to_str(extractor, schema, expected):
    assert extractor._schema_type_to_str(schema) == expected


@pytest.mark.parametrize(
    "info, expected_type",
    [
        ({"schema": {"type": "str"}}, "str"),
        ({"schema": {"type": "nullable", "schema": {"type": "int"}}}, "int"),
        ({"schema": {"type": "default", "schema": {"type": "list", "items_schema": {"type": "str"}}}}, "list[str]"),
    ]
)
def test_extract_param_type(extractor, info, expected_type):
    assert extractor._extract_param_type(info) == expected_type


def test_extract_tool_info(extractor):
    with mock.patch("generate_tool_specs.dir", return_value=["MockTool"]), \
         mock.patch("generate_tool_specs.getattr", return_value=MockTool):
        extractor.extract_all_tools()

        assert len(extractor.tools_spec) == 1
        tool_info = extractor.tools_spec[0]

        assert tool_info["name"] == "MockTool"
        assert tool_info["humanized_name"] == "Mock Search Tool"
        assert tool_info["description"] == "A tool that mocks search functionality"

        assert len(tool_info["env_vars"]) == 2
        api_key_var, rate_limit_var = tool_info["env_vars"]

        assert api_key_var["name"] == "SERPER_API_KEY"
        assert api_key_var["description"] == "API key for Serper"
        assert api_key_var["required"] == True
        assert api_key_var["default"] == None

        assert rate_limit_var["name"] == "API_RATE_LIMIT"
        assert rate_limit_var["description"] == "API rate limit"
        assert rate_limit_var["required"] == False
        assert rate_limit_var["default"] == "100"

        assert len(tool_info["run_params"]) == 3

        params = {p["name"]: p for p in tool_info["run_params"]}
        assert params["query"]["description"] == "The query parameter"
        assert params["query"]["type"] == "str"

        assert params["count"]["description"] == "Number of results to return"
        assert params["count"]["type"] == "int"

        assert params["filters"]["description"] == ""
        assert params["filters"]["type"] == "list[str]"


def test_save_to_json(extractor, tmp_path):
    extractor.tools_spec = [{
        "name": "TestTool",
        "humanized_name": "Test Tool",
        "description": "A test tool",
        "run_params": [
            {"name": "param1", "description": "Test parameter", "type": "str"}
        ]
    }]

    file_path = tmp_path / "output.json"
    extractor.save_to_json(str(file_path))

    assert file_path.exists()

    with open(file_path, "r") as f:
        data = json.load(f)

    assert "tools" in data
    assert len(data["tools"]) == 1
    assert data["tools"][0]["humanized_name"] == "Test Tool"
    assert data["tools"][0]["run_params"][0]["name"] == "param1"


@pytest.mark.integration
def test_full_extraction_process():
    extractor = ToolSpecExtractor()
    specs = extractor.extract_all_tools()

    assert len(specs) > 0

    for tool in specs:
        assert "name" in tool
        assert "humanized_name" in tool and tool["humanized_name"]
        assert "description" in tool
        assert isinstance(tool["run_params"], list)
        for param in tool["run_params"]:
            assert "name" in param and param["name"]
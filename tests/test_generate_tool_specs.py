import json
from typing import List, Optional, Type

import pytest
from pydantic import BaseModel, Field
from unittest import mock

from generate_tool_specs import ToolSpecExtractor
from crewai.tools.base_tool import BaseTool, EnvVar

class MockToolSchema(BaseModel):
    query: str = Field(..., description="The query parameter")
    count: int = Field(5, description="Number of results to return")
    filters: Optional[List[str]] = Field(None, description="Optional filters to apply")


class MockTool(BaseTool):
    name: str = "Mock Search Tool"
    description: str = "A tool that mocks search functionality"
    args_schema: Type[BaseModel] = MockToolSchema

    another_parameter: str = Field("Another way to define a default value", description="")
    my_parameter: str = Field("This is default value", description="What a description")
    my_parameter_bool: bool = Field(False)
    package_dependencies: List[str] = Field(["this-is-a-required-package", "another-required-package"], description="")
    env_vars: List[EnvVar] = [
        EnvVar(name="SERPER_API_KEY", description="API key for Serper", required=True, default=None),
        EnvVar(name="API_RATE_LIMIT", description="API rate limit", required=False, default="100")
    ]

@pytest.fixture
def extractor():
    ext = ToolSpecExtractor()
    return ext


def test_unwrap_schema(extractor):
    nested_schema = {
        "type": "function-after",
        "schema": {"type": "default", "schema": {"type": "str", "value": "test"}}
    }
    result = extractor._unwrap_schema(nested_schema)
    assert result["type"] == "str"
    assert result["value"] == "test"


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


def test_extract_all_tools(extractor):
    with mock.patch("generate_tool_specs.dir", return_value=["MockTool"]), \
         mock.patch("generate_tool_specs.getattr", return_value=MockTool):
        extractor.extract_all_tools()

        assert len(extractor.tools_spec) == 1
        tool_info = extractor.tools_spec[0]

        assert tool_info.keys() == {
            "name",
            "humanized_name",
            "description",
            "run_params",
            "env_vars",
            "init_params",
            "package_dependencies",
        }

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
        assert params["query"]["default"] == ""

        assert params["count"]["type"] == "int"
        assert params["count"]["default"] == 5

        assert params["filters"]["description"] == "Optional filters to apply"
        assert params["filters"]["type"] == "list[str]"
        assert params["filters"]["default"] == ""

        assert tool_info["package_dependencies"] == ["this-is-a-required-package", "another-required-package"]


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
import json
from unittest import mock

from crewai.tools.base_tool import BaseTool, EnvVar
from crewai_tools.generate_tool_specs import ToolSpecExtractor
from pydantic import BaseModel, Field
import pytest


class MockToolSchema(BaseModel):
    query: str = Field(..., description="The query parameter")
    count: int = Field(5, description="Number of results to return")
    filters: list[str] | None = Field(None, description="Optional filters to apply")


class MockTool(BaseTool):
    name: str = "Mock Search Tool"
    description: str = "A tool that mocks search functionality"
    args_schema: type[BaseModel] = MockToolSchema

    another_parameter: str = Field(
        "Another way to define a default value", description=""
    )
    my_parameter: str = Field("This is default value", description="What a description")
    my_parameter_bool: bool = Field(False)
    package_dependencies: list[str] = Field(
        ["this-is-a-required-package", "another-required-package"], description=""
    )
    env_vars: list[EnvVar] = [
        EnvVar(
            name="SERPER_API_KEY",
            description="API key for Serper",
            required=True,
            default=None,
        ),
        EnvVar(
            name="API_RATE_LIMIT",
            description="API rate limit",
            required=False,
            default="100",
        ),
    ]


@pytest.fixture
def extractor():
    ext = ToolSpecExtractor()
    return ext


def test_unwrap_schema(extractor):
    nested_schema = {
        "type": "function-after",
        "schema": {"type": "default", "schema": {"type": "str", "value": "test"}},
    }
    result = extractor._unwrap_schema(nested_schema)
    assert result["type"] == "str"
    assert result["value"] == "test"


@pytest.fixture
def mock_tool_extractor(extractor):
    with (
        mock.patch("crewai_tools.generate_tool_specs.dir", return_value=["MockTool"]),
        mock.patch("crewai_tools.generate_tool_specs.getattr", return_value=MockTool),
    ):
        extractor.extract_all_tools()
        assert len(extractor.tools_spec) == 1
        return extractor.tools_spec[0]


def test_extract_basic_tool_info(mock_tool_extractor):
    tool_info = mock_tool_extractor

    assert tool_info.keys() == {
        "name",
        "humanized_name",
        "description",
        "run_params_schema",
        "env_vars",
        "init_params_schema",
        "package_dependencies",
    }

    assert tool_info["name"] == "MockTool"
    assert tool_info["humanized_name"] == "Mock Search Tool"
    assert tool_info["description"] == "A tool that mocks search functionality"


def test_extract_init_params_schema(mock_tool_extractor):
    tool_info = mock_tool_extractor
    init_params_schema = tool_info["init_params_schema"]

    assert init_params_schema.keys() == {
        "$defs",
        "properties",
        "title",
        "type",
    }

    another_parameter = init_params_schema["properties"]["another_parameter"]
    assert another_parameter["description"] == ""
    assert another_parameter["default"] == "Another way to define a default value"
    assert another_parameter["type"] == "string"

    my_parameter = init_params_schema["properties"]["my_parameter"]
    assert my_parameter["description"] == "What a description"
    assert my_parameter["default"] == "This is default value"
    assert my_parameter["type"] == "string"

    my_parameter_bool = init_params_schema["properties"]["my_parameter_bool"]
    assert not my_parameter_bool["default"]
    assert my_parameter_bool["type"] == "boolean"


def test_extract_env_vars(mock_tool_extractor):
    tool_info = mock_tool_extractor

    assert len(tool_info["env_vars"]) == 2
    api_key_var, rate_limit_var = tool_info["env_vars"]
    assert api_key_var["name"] == "SERPER_API_KEY"
    assert api_key_var["description"] == "API key for Serper"
    assert api_key_var["required"]
    assert api_key_var["default"] is None

    assert rate_limit_var["name"] == "API_RATE_LIMIT"
    assert rate_limit_var["description"] == "API rate limit"
    assert not rate_limit_var["required"]
    assert rate_limit_var["default"] == "100"


def test_extract_run_params_schema(mock_tool_extractor):
    tool_info = mock_tool_extractor

    run_params_schema = tool_info["run_params_schema"]
    assert run_params_schema.keys() == {
        "properties",
        "required",
        "title",
        "type",
    }

    query_param = run_params_schema["properties"]["query"]
    assert query_param["description"] == "The query parameter"
    assert query_param["type"] == "string"

    count_param = run_params_schema["properties"]["count"]
    assert count_param["type"] == "integer"
    assert count_param["default"] == 5

    filters_param = run_params_schema["properties"]["filters"]
    assert filters_param["description"] == "Optional filters to apply"
    assert filters_param["default"] is None
    assert filters_param["anyOf"] == [
        {"items": {"type": "string"}, "type": "array"},
        {"type": "null"},
    ]


def test_extract_package_dependencies(mock_tool_extractor):
    tool_info = mock_tool_extractor
    assert tool_info["package_dependencies"] == [
        "this-is-a-required-package",
        "another-required-package",
    ]


def test_save_to_json(extractor, tmp_path):
    extractor.tools_spec = [
        {
            "name": "TestTool",
            "humanized_name": "Test Tool",
            "description": "A test tool",
            "run_params_schema": [
                {"name": "param1", "description": "Test parameter", "type": "str"}
            ],
        }
    ]

    file_path = tmp_path / "output.json"
    extractor.save_to_json(str(file_path))

    assert file_path.exists()

    with open(file_path, "r") as f:
        data = json.load(f)

    assert "tools" in data
    assert len(data["tools"]) == 1
    assert data["tools"][0]["humanized_name"] == "Test Tool"
    assert data["tools"][0]["run_params_schema"][0]["name"] == "param1"

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
    # Use default_factory like real tools do (not direct default)
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["this-is-a-required-package", "another-required-package"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
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
    )


# --- Intermediate base class (like RagTool, BraveSearchToolBase) ---
class MockIntermediateBase(BaseTool):
    """Simulates an intermediate tool base class (e.g. RagTool, BraveSearchToolBase)."""

    name: str = "Intermediate Base"
    description: str = "An intermediate tool base"
    shared_config: str = Field("default_config", description="Config from intermediate base")

    def _run(self, query: str) -> str:
        return query


class MockDerivedTool(MockIntermediateBase):
    """A tool inheriting from an intermediate base, like CodeDocsSearchTool(RagTool)."""

    name: str = "Derived Tool"
    description: str = "A tool that inherits from intermediate base"
    derived_param: str = Field("derived_default", description="Param specific to derived tool")


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
        "required",
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


def test_base_tool_fields_excluded_from_init_params(mock_tool_extractor):
    """BaseTool internal fields (including computed_field like tool_type) must
    never appear in init_params_schema. Studio reads this schema to render
    the tool config UI — internal fields confuse users."""
    init_schema = mock_tool_extractor["init_params_schema"]
    props = set(init_schema.get("properties", {}).keys())
    required = set(init_schema.get("required", []))

    # These are all BaseTool's own fields — none should leak
    base_fields = {"name", "description", "env_vars", "args_schema",
                   "description_updated", "cache_function", "result_as_answer",
                   "max_usage_count", "current_usage_count", "tool_type",
                   "package_dependencies"}

    leaked_props = base_fields & props
    assert not leaked_props, (
        f"BaseTool fields leaked into init_params_schema properties: {leaked_props}"
    )
    leaked_required = base_fields & required
    assert not leaked_required, (
        f"BaseTool fields leaked into init_params_schema required: {leaked_required}"
    )


def test_intermediate_base_fields_preserved_for_derived_tool(extractor):
    """When a tool inherits from an intermediate base (e.g. RagTool),
    the intermediate's fields should be included — only BaseTool's own
    fields are excluded."""
    with (
        mock.patch(
            "crewai_tools.generate_tool_specs.dir",
            return_value=["MockDerivedTool"],
        ),
        mock.patch(
            "crewai_tools.generate_tool_specs.getattr",
            return_value=MockDerivedTool,
        ),
    ):
        extractor.extract_all_tools()
        assert len(extractor.tools_spec) == 1
        tool_info = extractor.tools_spec[0]

    props = set(tool_info["init_params_schema"].get("properties", {}).keys())

    # Intermediate base's field should be preserved
    assert "shared_config" in props, (
        "Intermediate base class fields should be preserved in init_params_schema"
    )
    # Derived tool's own field should be preserved
    assert "derived_param" in props, (
        "Derived tool's own fields should be preserved in init_params_schema"
    )
    # BaseTool internals should still be excluded
    assert "tool_type" not in props
    assert "cache_function" not in props
    assert "result_as_answer" not in props


def test_future_base_tool_field_auto_excluded(extractor):
    """If a new field is added to BaseTool in the future, it should be
    automatically excluded from spec generation without needing to update
    the ignored list. This test verifies the allowlist approach works
    by checking that ONLY non-BaseTool fields appear."""
    with (
        mock.patch("crewai_tools.generate_tool_specs.dir", return_value=["MockTool"]),
        mock.patch("crewai_tools.generate_tool_specs.getattr", return_value=MockTool),
    ):
        extractor.extract_all_tools()
        tool_info = extractor.tools_spec[0]

    props = set(tool_info["init_params_schema"].get("properties", {}).keys())
    base_all = set(BaseTool.model_fields) | set(BaseTool.model_computed_fields)

    leaked = base_all & props
    assert not leaked, (
        f"BaseTool fields should be auto-excluded but found: {leaked}. "
        "The spec generator should dynamically compute BaseTool's fields "
        "instead of using a hardcoded denylist."
    )


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

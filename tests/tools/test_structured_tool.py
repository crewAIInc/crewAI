
import pytest
from pydantic import BaseModel, Field

from crewai.tools.structured_tool import CrewStructuredTool


# Test fixtures
@pytest.fixture
def basic_function():
    def test_func(param1: str, param2: int = 0) -> str:
        """Test function with basic params."""
        return f"{param1} {param2}"

    return test_func


@pytest.fixture
def schema_class():
    class TestSchema(BaseModel):
        param1: str
        param2: int = Field(default=0)

    return TestSchema


class InternalCrewStructuredTool:
    def test_initialization(self, basic_function, schema_class) -> None:
        """Test basic initialization of CrewStructuredTool."""
        tool = CrewStructuredTool(
            name="test_tool",
            description="Test tool description",
            func=basic_function,
            args_schema=schema_class,
        )

        assert tool.name == "test_tool"
        assert tool.description == "Test tool description"
        assert tool.func == basic_function
        assert tool.args_schema == schema_class

    def test_from_function(self, basic_function) -> None:
        """Test creating tool from function."""
        tool = CrewStructuredTool.from_function(
            func=basic_function, name="test_tool", description="Test description",
        )

        assert tool.name == "test_tool"
        assert tool.description == "Test description"
        assert tool.func == basic_function
        assert isinstance(tool.args_schema, type(BaseModel))

    def test_validate_function_signature(self, basic_function, schema_class) -> None:
        """Test function signature validation."""
        tool = CrewStructuredTool(
            name="test_tool",
            description="Test tool",
            func=basic_function,
            args_schema=schema_class,
        )

        # Should not raise any exceptions
        tool._validate_function_signature()

    @pytest.mark.asyncio
    async def test_ainvoke(self, basic_function) -> None:
        """Test asynchronous invocation."""
        tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

        result = await tool.ainvoke(input={"param1": "test"})
        assert result == "test 0"

    def test_parse_args_dict(self, basic_function) -> None:
        """Test parsing dictionary arguments."""
        tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

        parsed = tool._parse_args({"param1": "test", "param2": 42})
        assert parsed["param1"] == "test"
        assert parsed["param2"] == 42

    def test_parse_args_string(self, basic_function) -> None:
        """Test parsing string arguments."""
        tool = CrewStructuredTool.from_function(func=basic_function, name="test_tool")

        parsed = tool._parse_args('{"param1": "test", "param2": 42}')
        assert parsed["param1"] == "test"
        assert parsed["param2"] == 42

    def test_complex_types(self) -> None:
        """Test handling of complex parameter types."""

        def complex_func(nested: dict, items: list) -> str:
            """Process complex types."""
            return f"Processed {len(items)} items with {len(nested)} nested keys"

        tool = CrewStructuredTool.from_function(
            func=complex_func, name="test_tool", description="Test complex types",
        )
        result = tool.invoke({"nested": {"key": "value"}, "items": [1, 2, 3]})
        assert result == "Processed 3 items with 1 nested keys"

    def test_schema_inheritance(self) -> None:
        """Test tool creation with inherited schema."""

        def extended_func(base_param: str, extra_param: int) -> str:
            """Test function with inherited schema."""
            return f"{base_param} {extra_param}"

        class BaseSchema(BaseModel):
            base_param: str

        class ExtendedSchema(BaseSchema):
            extra_param: int

        tool = CrewStructuredTool.from_function(
            func=extended_func, name="test_tool", args_schema=ExtendedSchema,
        )

        result = tool.invoke({"base_param": "test", "extra_param": 42})
        assert result == "test 42"

    def test_default_values_in_schema(self) -> None:
        """Test handling of default values in schema."""

        def default_func(
            required_param: str,
            optional_param: str = "default",
            nullable_param: int | None = None,
        ) -> str:
            """Test function with default values."""
            return f"{required_param} {optional_param} {nullable_param}"

        tool = CrewStructuredTool.from_function(
            func=default_func, name="test_tool", description="Test defaults",
        )

        # Test with minimal parameters
        result = tool.invoke({"required_param": "test"})
        assert result == "test default None"

        # Test with all parameters
        result = tool.invoke(
            {"required_param": "test", "optional_param": "custom", "nullable_param": 42},
        )
        assert result == "test custom 42"

import os
import tempfile
from pathlib import Path

import pytest
from crewai.utilities import project_utils as utils


def create_file(path, content):
    with open(path, "w") as f:
        f.write(content)


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing tool extraction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def create_init_file(directory, content):
    return create_file(directory / "__init__.py", content)


def test_extract_available_exports_empty_project(temp_project_dir, capsys):
    with pytest.raises(SystemExit):
        utils.extract_available_exports(dir_path=temp_project_dir)
    captured = capsys.readouterr()

    assert "No valid tools were exposed in your __init__.py file" in captured.out


def test_extract_available_exports_no_init_file(temp_project_dir, capsys):
    (temp_project_dir / "some_file.py").write_text("print('hello')")
    with pytest.raises(SystemExit):
        utils.extract_available_exports(dir_path=temp_project_dir)
    captured = capsys.readouterr()

    assert "No valid tools were exposed in your __init__.py file" in captured.out


def test_extract_available_exports_empty_init_file(temp_project_dir, capsys):
    create_init_file(temp_project_dir, "")
    with pytest.raises(SystemExit):
        utils.extract_available_exports(dir_path=temp_project_dir)
    captured = capsys.readouterr()

    assert "Warning: No __all__ defined in" in captured.out


def test_extract_available_exports_no_all_variable(temp_project_dir, capsys):
    create_init_file(
        temp_project_dir,
        "from crewai.tools import BaseTool\n\nclass MyTool(BaseTool):\n    pass",
    )
    with pytest.raises(SystemExit):
        utils.extract_available_exports(dir_path=temp_project_dir)
    captured = capsys.readouterr()

    assert "Warning: No __all__ defined in" in captured.out


def test_extract_available_exports_valid_base_tool_class(temp_project_dir):
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"

__all__ = ['MyTool']
""",
    )
    tools = utils.extract_available_exports(dir_path=temp_project_dir)
    assert [{"name": "MyTool"}] == tools


def test_extract_available_exports_valid_tool_decorator(temp_project_dir):
    create_init_file(
        temp_project_dir,
        """from crewai.tools import tool

@tool
def my_tool_function(text: str) -> str:
    \"\"\"A test tool function\"\"\"
    return text

__all__ = ['my_tool_function']
""",
    )
    tools = utils.extract_available_exports(dir_path=temp_project_dir)
    assert [{"name": "my_tool_function"}] == tools


def test_extract_available_exports_multiple_valid_tools(temp_project_dir):
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool, tool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"

@tool
def my_tool_function(text: str) -> str:
    \"\"\"A test tool function\"\"\"
    return text

__all__ = ['MyTool', 'my_tool_function']
""",
    )
    tools = utils.extract_available_exports(dir_path=temp_project_dir)
    assert [{"name": "MyTool"}, {"name": "my_tool_function"}] == tools


def test_extract_available_exports_with_invalid_tool_decorator(temp_project_dir):
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"

def not_a_tool():
    pass

__all__ = ['MyTool', 'not_a_tool']
""",
    )
    tools = utils.extract_available_exports(dir_path=temp_project_dir)
    assert [{"name": "MyTool"}] == tools


def test_extract_available_exports_import_error(temp_project_dir, capsys):
    create_init_file(
        temp_project_dir,
        """from nonexistent_module import something

class MyTool(BaseTool):
    pass

__all__ = ['MyTool']
""",
    )
    with pytest.raises(SystemExit):
        utils.extract_available_exports(dir_path=temp_project_dir)
    captured = capsys.readouterr()

    assert "nonexistent_module" in captured.out


def test_extract_available_exports_syntax_error(temp_project_dir, capsys):
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    # Missing closing parenthesis
    def __init__(self, name:
        pass

__all__ = ['MyTool']
""",
    )
    with pytest.raises(SystemExit):
        utils.extract_available_exports(dir_path=temp_project_dir)
    captured = capsys.readouterr()

    assert "was never closed" in captured.out


@pytest.fixture
def mock_crew():
    from crewai.crew import Crew

    class MockCrew(Crew):
        def __init__(self):
            pass

    return MockCrew()


@pytest.fixture
def temp_crew_project():
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)

        crew_content = """
        from crewai.crew import Crew
        from crewai.agent import Agent

        def create_crew() -> Crew:
            agent = Agent(role="test", goal="test", backstory="test")
            return Crew(agents=[agent], tasks=[])

        # Direct crew instance
        direct_crew = Crew(agents=[], tasks=[])
        """

        with open("crew.py", "w") as f:
            f.write(crew_content)

        os.makedirs("src", exist_ok=True)
        with open(os.path.join("src", "crew.py"), "w") as f:
            f.write(crew_content)

        # Create a src/templates directory that should be ignored
        os.makedirs(os.path.join("src", "templates"), exist_ok=True)
        with open(os.path.join("src", "templates", "crew.py"), "w") as f:
            f.write("# This should be ignored")

        yield temp_dir

        os.chdir(old_cwd)


def test_get_crews_finds_valid_crews(temp_crew_project, monkeypatch, mock_crew):
    def mock_fetch_crews(module_attr):
        return [mock_crew]

    monkeypatch.setattr(utils, "fetch_crews", mock_fetch_crews)

    crews = utils.get_crews()

    assert len(crews) > 0
    assert mock_crew in crews


def test_get_crews_with_nonexistent_file(temp_crew_project):
    crews = utils.get_crews(crew_path="nonexistent.py", require=False)
    assert len(crews) == 0


def test_get_crews_with_required_nonexistent_file(temp_crew_project, capsys):
    with pytest.raises(SystemExit):
        utils.get_crews(crew_path="nonexistent.py", require=True)

    captured = capsys.readouterr()
    assert "No valid Crew instance found" in captured.out


def test_get_crews_with_invalid_module(temp_crew_project, capsys):
    with open("crew.py", "w") as f:
        f.write("import nonexistent_module\n")

    crews = utils.get_crews(crew_path="crew.py", require=False)
    assert len(crews) == 0

    with pytest.raises(SystemExit):
        utils.get_crews(crew_path="crew.py", require=True)

    captured = capsys.readouterr()
    assert "Error" in captured.out


def test_get_crews_ignores_template_directories(
    temp_crew_project, monkeypatch, mock_crew
):
    template_crew_detected = False

    def mock_fetch_crews(module_attr):
        nonlocal template_crew_detected
        if hasattr(module_attr, "__file__") and "templates" in module_attr.__file__:
            template_crew_detected = True
        return [mock_crew]

    monkeypatch.setattr(utils, "fetch_crews", mock_fetch_crews)

    utils.get_crews()

    assert not template_crew_detected


# Tests for extract_tools_metadata


def test_extract_tools_metadata_empty_project(temp_project_dir):
    """Test that extract_tools_metadata returns empty list for empty project."""
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert metadata == []


def test_extract_tools_metadata_no_init_file(temp_project_dir):
    """Test that extract_tools_metadata returns empty list when no __init__.py exists."""
    (temp_project_dir / "some_file.py").write_text("print('hello')")
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert metadata == []


def test_extract_tools_metadata_empty_init_file(temp_project_dir):
    """Test that extract_tools_metadata returns empty list for empty __init__.py."""
    create_init_file(temp_project_dir, "")
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert metadata == []


def test_extract_tools_metadata_no_all_variable(temp_project_dir):
    """Test that extract_tools_metadata returns empty list when __all__ is not defined."""
    create_init_file(
        temp_project_dir,
        "from crewai.tools import BaseTool\n\nclass MyTool(BaseTool):\n    pass",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert metadata == []


def test_extract_tools_metadata_valid_base_tool_class(temp_project_dir):
    """Test that extract_tools_metadata extracts metadata from a valid BaseTool class."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"

__all__ = ['MyTool']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 1
    assert metadata[0]["name"] == "MyTool"
    assert metadata[0]["humanized_name"] == "my_tool"
    assert metadata[0]["description"] == "A test tool"


def test_extract_tools_metadata_with_args_schema(temp_project_dir):
    """Test that extract_tools_metadata extracts run_params_schema from args_schema."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool
from pydantic import BaseModel

class MyToolInput(BaseModel):
    query: str
    limit: int = 10

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"
    args_schema: type[BaseModel] = MyToolInput

__all__ = ['MyTool']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 1
    assert metadata[0]["name"] == "MyTool"
    run_params = metadata[0]["run_params_schema"]
    assert "properties" in run_params
    assert "query" in run_params["properties"]
    assert "limit" in run_params["properties"]


def test_extract_tools_metadata_with_env_vars(temp_project_dir):
    """Test that extract_tools_metadata extracts env_vars."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool
from crewai.tools.base_tool import EnvVar

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"
    env_vars: list[EnvVar] = [
        EnvVar(name="MY_API_KEY", description="API key for service", required=True),
        EnvVar(name="MY_OPTIONAL_VAR", description="Optional var", required=False, default="default_value"),
    ]

__all__ = ['MyTool']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 1
    env_vars = metadata[0]["env_vars"]
    assert len(env_vars) == 2
    assert env_vars[0]["name"] == "MY_API_KEY"
    assert env_vars[0]["description"] == "API key for service"
    assert env_vars[0]["required"] is True
    assert env_vars[1]["name"] == "MY_OPTIONAL_VAR"
    assert env_vars[1]["required"] is False
    assert env_vars[1]["default"] == "default_value"


def test_extract_tools_metadata_with_env_vars_field_default_factory(temp_project_dir):
    """Test that extract_tools_metadata extracts env_vars declared with Field(default_factory=...)."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool
from crewai.tools.base_tool import EnvVar
from pydantic import Field

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(name="MY_TOOL_API", description="API token for my tool", required=True),
        ]
    )

__all__ = ['MyTool']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 1
    env_vars = metadata[0]["env_vars"]
    assert len(env_vars) == 1
    assert env_vars[0]["name"] == "MY_TOOL_API"
    assert env_vars[0]["description"] == "API token for my tool"
    assert env_vars[0]["required"] is True


def test_extract_tools_metadata_with_custom_init_params(temp_project_dir):
    """Test that extract_tools_metadata extracts init_params_schema with custom params."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"
    api_endpoint: str = "https://api.example.com"
    timeout: int = 30

__all__ = ['MyTool']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 1
    init_params = metadata[0]["init_params_schema"]
    assert "properties" in init_params
    # Custom params should be included
    assert "api_endpoint" in init_params["properties"]
    assert "timeout" in init_params["properties"]
    # Base params should be filtered out
    assert "name" not in init_params["properties"]
    assert "description" not in init_params["properties"]


def test_extract_tools_metadata_multiple_tools(temp_project_dir):
    """Test that extract_tools_metadata extracts metadata from multiple tools."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class FirstTool(BaseTool):
    name: str = "first_tool"
    description: str = "First test tool"

class SecondTool(BaseTool):
    name: str = "second_tool"
    description: str = "Second test tool"

__all__ = ['FirstTool', 'SecondTool']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 2
    names = [m["name"] for m in metadata]
    assert "FirstTool" in names
    assert "SecondTool" in names


def test_extract_tools_metadata_multiple_init_files(temp_project_dir):
    """Test that extract_tools_metadata extracts metadata from multiple __init__.py files."""
    # Create tool in root __init__.py
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class RootTool(BaseTool):
    name: str = "root_tool"
    description: str = "Root tool"

__all__ = ['RootTool']
""",
    )

    # Create nested package with another tool
    nested_dir = temp_project_dir / "nested"
    nested_dir.mkdir()
    create_init_file(
        nested_dir,
        """from crewai.tools import BaseTool

class NestedTool(BaseTool):
    name: str = "nested_tool"
    description: str = "Nested tool"

__all__ = ['NestedTool']
""",
    )

    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 2
    names = [m["name"] for m in metadata]
    assert "RootTool" in names
    assert "NestedTool" in names


def test_extract_tools_metadata_ignores_non_tool_exports(temp_project_dir):
    """Test that extract_tools_metadata ignores non-BaseTool exports."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "A test tool"

def not_a_tool():
    pass

SOME_CONSTANT = "value"

__all__ = ['MyTool', 'not_a_tool', 'SOME_CONSTANT']
""",
    )
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert len(metadata) == 1
    assert metadata[0]["name"] == "MyTool"


def test_extract_tools_metadata_import_error_returns_empty(temp_project_dir):
    """Test that extract_tools_metadata returns empty list on import error."""
    create_init_file(
        temp_project_dir,
        """from nonexistent_module import something

class MyTool(BaseTool):
    pass

__all__ = ['MyTool']
""",
    )
    # Should not raise, just return empty list
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert metadata == []


def test_extract_tools_metadata_syntax_error_returns_empty(temp_project_dir):
    """Test that extract_tools_metadata returns empty list on syntax error."""
    create_init_file(
        temp_project_dir,
        """from crewai.tools import BaseTool

class MyTool(BaseTool):
    # Missing closing parenthesis
    def __init__(self, name:
        pass

__all__ = ['MyTool']
""",
    )
    # Should not raise, just return empty list
    metadata = utils.extract_tools_metadata(dir_path=str(temp_project_dir))
    assert metadata == []

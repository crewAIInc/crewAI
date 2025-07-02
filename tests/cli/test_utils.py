import os
import shutil
import tempfile
from pathlib import Path

import pytest

from crewai.cli import utils


@pytest.fixture
def temp_tree():
    root_dir = tempfile.mkdtemp()

    create_file(os.path.join(root_dir, "file1.txt"), "Hello, world!")
    create_file(os.path.join(root_dir, "file2.txt"), "Another file")
    os.mkdir(os.path.join(root_dir, "empty_dir"))
    nested_dir = os.path.join(root_dir, "nested_dir")
    os.mkdir(nested_dir)
    create_file(os.path.join(nested_dir, "nested_file.txt"), "Nested content")

    yield root_dir

    shutil.rmtree(root_dir)


def create_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def test_tree_find_and_replace_file_content(temp_tree):
    utils.tree_find_and_replace(temp_tree, "world", "universe")
    with open(os.path.join(temp_tree, "file1.txt"), "r") as f:
        assert f.read() == "Hello, universe!"


def test_tree_find_and_replace_file_name(temp_tree):
    old_path = os.path.join(temp_tree, "file2.txt")
    new_path = os.path.join(temp_tree, "file2_renamed.txt")
    os.rename(old_path, new_path)
    utils.tree_find_and_replace(temp_tree, "renamed", "modified")
    assert os.path.exists(os.path.join(temp_tree, "file2_modified.txt"))
    assert not os.path.exists(new_path)


def test_tree_find_and_replace_directory_name(temp_tree):
    utils.tree_find_and_replace(temp_tree, "empty", "renamed")
    assert os.path.exists(os.path.join(temp_tree, "renamed_dir"))
    assert not os.path.exists(os.path.join(temp_tree, "empty_dir"))


def test_tree_find_and_replace_nested_content(temp_tree):
    utils.tree_find_and_replace(temp_tree, "Nested", "Updated")
    with open(os.path.join(temp_tree, "nested_dir", "nested_file.txt"), "r") as f:
        assert f.read() == "Updated content"


def test_tree_find_and_replace_no_matches(temp_tree):
    utils.tree_find_and_replace(temp_tree, "nonexistent", "replacement")
    assert set(os.listdir(temp_tree)) == {
        "file1.txt",
        "file2.txt",
        "empty_dir",
        "nested_dir",
    }


def test_tree_copy_full_structure(temp_tree):
    dest_dir = tempfile.mkdtemp()
    try:
        utils.tree_copy(temp_tree, dest_dir)
        assert set(os.listdir(dest_dir)) == set(os.listdir(temp_tree))
        assert os.path.isfile(os.path.join(dest_dir, "file1.txt"))
        assert os.path.isfile(os.path.join(dest_dir, "file2.txt"))
        assert os.path.isdir(os.path.join(dest_dir, "empty_dir"))
        assert os.path.isdir(os.path.join(dest_dir, "nested_dir"))
        assert os.path.isfile(os.path.join(dest_dir, "nested_dir", "nested_file.txt"))
    finally:
        shutil.rmtree(dest_dir)


def test_tree_copy_preserve_content(temp_tree):
    dest_dir = tempfile.mkdtemp()
    try:
        utils.tree_copy(temp_tree, dest_dir)
        with open(os.path.join(dest_dir, "file1.txt"), "r") as f:
            assert f.read() == "Hello, world!"
        with open(os.path.join(dest_dir, "nested_dir", "nested_file.txt"), "r") as f:
            assert f.read() == "Nested content"
    finally:
        shutil.rmtree(dest_dir)


def test_tree_copy_to_existing_directory(temp_tree):
    dest_dir = tempfile.mkdtemp()
    try:
        create_file(os.path.join(dest_dir, "existing_file.txt"), "I was here first")
        utils.tree_copy(temp_tree, dest_dir)
        assert os.path.isfile(os.path.join(dest_dir, "existing_file.txt"))
        assert os.path.isfile(os.path.join(dest_dir, "file1.txt"))
    finally:
        shutil.rmtree(dest_dir)


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


def test_get_crews_ignores_template_directories(temp_crew_project, monkeypatch, mock_crew):
    template_crew_detected = False

    def mock_fetch_crews(module_attr):
        nonlocal template_crew_detected
        if hasattr(module_attr, "__file__") and "templates" in module_attr.__file__:
            template_crew_detected = True
        return [mock_crew]

    monkeypatch.setattr(utils, "fetch_crews", mock_fetch_crews)

    utils.get_crews()

    assert not template_crew_detected

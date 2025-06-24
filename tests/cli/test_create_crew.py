import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.create_crew import create_crew, create_folder_structure


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dir():
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


def test_create_folder_structure_strips_single_trailing_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_path, folder_name, class_name = create_folder_structure("hello/")
        
        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()


def test_create_folder_structure_strips_multiple_trailing_slashes():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_path, folder_name, class_name = create_folder_structure("hello///")
        
        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()


def test_create_folder_structure_handles_complex_name_with_trailing_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_path, folder_name, class_name = create_folder_structure("my-awesome_project/")
        
        assert folder_name == "my_awesome_project"
        assert class_name == "MyAwesomeProject"
        assert folder_path.name == "my_awesome_project"
        assert folder_path.exists()


def test_create_folder_structure_normal_name_unchanged():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_path, folder_name, class_name = create_folder_structure("hello")
        
        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()





def test_create_folder_structure_with_parent_folder():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        parent_path = Path(temp_dir) / "parent"
        parent_path.mkdir()
        
        folder_path, folder_name, class_name = create_folder_structure("child/", parent_folder=parent_path)
        
        assert folder_name == "child"
        assert class_name == "Child"
        assert folder_path.name == "child"
        assert folder_path.parent == parent_path
        assert folder_path.exists()


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_with_trailing_slash_creates_valid_project(mock_load_env, mock_write_env, mock_copy_template, temp_dir):
    mock_load_env.return_value = {}
    
    with tempfile.TemporaryDirectory() as work_dir:
        os.chdir(work_dir)
        create_crew("test-project/", skip_provider=True)
        
        project_path = Path(work_dir) / "test_project"
        assert project_path.exists()
        assert (project_path / "src" / "test_project").exists()
        
        mock_copy_template.assert_called()
        copy_calls = mock_copy_template.call_args_list
        
        for call in copy_calls:
            args = call[0]
            if len(args) >= 5:
                folder_name_arg = args[4]
                assert not folder_name_arg.endswith("/"), f"folder_name should not end with slash: {folder_name_arg}"


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_with_multiple_trailing_slashes(mock_load_env, mock_write_env, mock_copy_template, temp_dir):
    mock_load_env.return_value = {}
    
    with tempfile.TemporaryDirectory() as work_dir:
        os.chdir(work_dir)
        create_crew("test-project///", skip_provider=True)
        
        project_path = Path(work_dir) / "test_project"
        assert project_path.exists()
        assert (project_path / "src" / "test_project").exists()


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_normal_name_still_works(mock_load_env, mock_write_env, mock_copy_template, temp_dir):
    mock_load_env.return_value = {}
    
    with tempfile.TemporaryDirectory() as work_dir:
        os.chdir(work_dir)
        create_crew("normal-project", skip_provider=True)
        
        project_path = Path(work_dir) / "normal_project"
        assert project_path.exists()
        assert (project_path / "src" / "normal_project").exists()


def test_create_folder_structure_handles_spaces_and_dashes_with_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        folder_path, folder_name, class_name = create_folder_structure("My Cool-Project/")
        
        assert folder_name == "my_cool_project"
        assert class_name == "MyCoolProject"
        assert folder_path.name == "my_cool_project"
        assert folder_path.exists()


def test_create_folder_structure_handles_invalid_class_name_edge_cases():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        folder_path, folder_name, class_name = create_folder_structure("123project/")
        assert folder_name == "123project"
        assert class_name == "Crew123Project"
        assert class_name.isidentifier()
        assert folder_path.exists()
        
        folder_path, folder_name, class_name = create_folder_structure("True/")
        assert folder_name == "true"
        assert class_name == "TrueCrew"
        assert class_name.isidentifier()
        assert folder_path.exists()
        
        folder_path, folder_name, class_name = create_folder_structure("   /")
        assert folder_name == "___"  # Spaces become underscores in folder_name
        assert class_name == "DefaultCrew"  # But class_name should be DefaultCrew for whitespace-only input
        assert class_name.isidentifier()
        assert folder_path.exists()
        
        folder_path, folder_name, class_name = create_folder_structure("hello@world/")
        assert folder_name == "hello@world"
        assert class_name == "HelloWorld"
        assert class_name.isidentifier()
        assert folder_path.exists()


def test_create_folder_structure_class_names_are_valid_python_identifiers():
    import keyword
    
    test_cases = [
        "hello/",
        "my-project/",
        "123project/",
        "class/",
        "def/",
        "import/",
        "True/",
        "False/",
        "None/",
        "hello.world/",
        "hello@world/",
        "hello#world/",
        "hello$world/",
        "///",
        "",
        "   /",
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        for i, test_case in enumerate(test_cases):
            unique_name = f"{test_case.rstrip('/')}_test_{i}/"
            if not test_case.strip('/'):
                unique_name = f"empty_test_{i}/"
                
            folder_path, folder_name, class_name = create_folder_structure(unique_name)
            
            assert class_name.isidentifier(), f"Class name '{class_name}' from input '{unique_name}' is not a valid identifier"
            assert not keyword.iskeyword(class_name), f"Class name '{class_name}' from input '{unique_name}' is a Python keyword"
            assert class_name not in ('True', 'False', 'None'), f"Class name '{class_name}' from input '{unique_name}' is a Python built-in"
            
            if folder_path.exists():
                shutil.rmtree(folder_path)


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_with_parent_folder_and_trailing_slash(mock_load_env, mock_write_env, mock_copy_template, temp_dir):
    mock_load_env.return_value = {}
    
    with tempfile.TemporaryDirectory() as work_dir:
        parent_path = Path(work_dir) / "parent"
        parent_path.mkdir()
        
        create_crew("child-crew/", skip_provider=True, parent_folder=parent_path)
        
        crew_path = parent_path / "child_crew"
        assert crew_path.exists()
        assert not (crew_path / "src").exists()

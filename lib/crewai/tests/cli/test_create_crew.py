import keyword
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
        folder_path, folder_name, class_name = create_folder_structure(
            "hello/", parent_folder=temp_dir
        )

        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_strips_multiple_trailing_slashes():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "hello///", parent_folder=temp_dir
        )

        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_handles_complex_name_with_trailing_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "my-awesome_project/", parent_folder=temp_dir
        )

        assert folder_name == "my_awesome_project"
        assert class_name == "MyAwesomeProject"
        assert folder_path.name == "my_awesome_project"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_normal_name_unchanged():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "hello", parent_folder=temp_dir
        )

        assert folder_name == "hello"
        assert class_name == "Hello"
        assert folder_path.name == "hello"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_with_parent_folder():
    with tempfile.TemporaryDirectory() as temp_dir:
        parent_path = Path(temp_dir) / "parent"
        parent_path.mkdir()

        folder_path, folder_name, class_name = create_folder_structure(
            "child/", parent_folder=parent_path
        )

        assert folder_name == "child"
        assert class_name == "Child"
        assert folder_path.name == "child"
        assert folder_path.parent == parent_path
        assert folder_path.exists()


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_with_trailing_slash_creates_valid_project(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        with mock.patch(
            "crewai.cli.create_crew.create_folder_structure"
        ) as mock_create_folder:
            mock_folder_path = Path(work_dir) / "test_project"
            mock_create_folder.return_value = (
                mock_folder_path,
                "test_project",
                "TestProject",
            )

            create_crew("test-project/", skip_provider=True)

            mock_create_folder.assert_called_once_with("test-project/", None)
            mock_copy_template.assert_called()
            copy_calls = mock_copy_template.call_args_list

            for call in copy_calls:
                args = call[0]
                if len(args) >= 5:
                    folder_name_arg = args[4]
                    assert not folder_name_arg.endswith("/"), (
                        f"folder_name should not end with slash: {folder_name_arg}"
                    )


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_with_multiple_trailing_slashes(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        with mock.patch(
            "crewai.cli.create_crew.create_folder_structure"
        ) as mock_create_folder:
            mock_folder_path = Path(work_dir) / "test_project"
            mock_create_folder.return_value = (
                mock_folder_path,
                "test_project",
                "TestProject",
            )

            create_crew("test-project///", skip_provider=True)

            mock_create_folder.assert_called_once_with("test-project///", None)


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_normal_name_still_works(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        with mock.patch(
            "crewai.cli.create_crew.create_folder_structure"
        ) as mock_create_folder:
            mock_folder_path = Path(work_dir) / "normal_project"
            mock_create_folder.return_value = (
                mock_folder_path,
                "normal_project",
                "NormalProject",
            )

            create_crew("normal-project", skip_provider=True)

            mock_create_folder.assert_called_once_with("normal-project", None)


def test_create_folder_structure_handles_spaces_and_dashes_with_slash():
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_path, folder_name, class_name = create_folder_structure(
            "My Cool-Project/", parent_folder=temp_dir
        )

        assert folder_name == "my_cool_project"
        assert class_name == "MyCoolProject"
        assert folder_path.name == "my_cool_project"
        assert folder_path.exists()
        assert folder_path.parent == Path(temp_dir)


def test_create_folder_structure_raises_error_for_invalid_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        invalid_cases = [
            ("123project/", "cannot start with a digit"),
            ("True/", "reserved Python keyword"),
            ("False/", "reserved Python keyword"),
            ("None/", "reserved Python keyword"),
            ("class/", "reserved Python keyword"),
            ("def/", "reserved Python keyword"),
            ("   /", "empty or contain only whitespace"),
            ("", "empty or contain only whitespace"),
            ("@#$/", "contains no valid characters"),
        ]

        for invalid_name, expected_error in invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                create_folder_structure(invalid_name, parent_folder=temp_dir)


def test_create_folder_structure_validates_names():
    with tempfile.TemporaryDirectory() as temp_dir:
        valid_cases = [
            ("hello/", "hello", "Hello"),
            ("my-project/", "my_project", "MyProject"),
            ("hello_world/", "hello_world", "HelloWorld"),
            ("valid123/", "valid123", "Valid123"),
            ("hello.world/", "helloworld", "HelloWorld"),
            ("hello@world/", "helloworld", "HelloWorld"),
        ]

        for valid_name, expected_folder, expected_class in valid_cases:
            folder_path, folder_name, class_name = create_folder_structure(
                valid_name, parent_folder=temp_dir
            )
            assert folder_name == expected_folder
            assert class_name == expected_class

            assert folder_name.isidentifier(), (
                f"folder_name '{folder_name}' should be valid Python identifier"
            )
            assert not keyword.iskeyword(folder_name), (
                f"folder_name '{folder_name}' should not be Python keyword"
            )
            assert not folder_name[0].isdigit(), (
                f"folder_name '{folder_name}' should not start with digit"
            )

            assert class_name.isidentifier(), (
                f"class_name '{class_name}' should be valid Python identifier"
            )
            assert not keyword.iskeyword(class_name), (
                f"class_name '{class_name}' should not be Python keyword"
            )
            assert folder_path.parent == Path(temp_dir)

            if folder_path.exists():
                shutil.rmtree(folder_path)


@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
def test_create_crew_with_parent_folder_and_trailing_slash(
    mock_load_env, mock_write_env, mock_copy_template, temp_dir
):
    mock_load_env.return_value = {}

    with tempfile.TemporaryDirectory() as work_dir:
        parent_path = Path(work_dir) / "parent"
        parent_path.mkdir()

        create_crew("child-crew/", skip_provider=True, parent_folder=parent_path)

        crew_path = parent_path / "child_crew"
        assert crew_path.exists()
        assert not (crew_path / "src").exists()


def test_create_folder_structure_folder_name_validation():
    """Test that folder names are validated as valid Python module names"""
    with tempfile.TemporaryDirectory() as temp_dir:
        folder_invalid_cases = [
            ("123invalid/", "cannot start with a digit.*invalid Python module name"),
            ("import/", "reserved Python keyword"),
            ("class/", "reserved Python keyword"),
            ("for/", "reserved Python keyword"),
            ("@#$invalid/", "contains no valid characters.*Python module name"),
        ]

        for invalid_name, expected_error in folder_invalid_cases:
            with pytest.raises(ValueError, match=expected_error):
                create_folder_structure(invalid_name, parent_folder=temp_dir)

        valid_cases = [
            ("hello-world/", "hello_world"),
            ("my.project/", "myproject"),
            ("test@123/", "test123"),
            ("valid_name/", "valid_name"),
        ]

        for valid_name, expected_folder in valid_cases:
            folder_path, folder_name, class_name = create_folder_structure(
                valid_name, parent_folder=temp_dir
            )
            assert folder_name == expected_folder
            assert folder_name.isidentifier()
            assert not keyword.iskeyword(folder_name)

            if folder_path.exists():
                shutil.rmtree(folder_path)


def test_create_folder_structure_rejects_reserved_names():
    """Test that reserved script names are rejected to prevent pyproject.toml conflicts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reserved_names = ["test", "train", "replay", "run_crew", "run_with_trigger"]

        for reserved_name in reserved_names:
            with pytest.raises(ValueError, match="which is reserved"):
                create_folder_structure(reserved_name, parent_folder=temp_dir)

            with pytest.raises(ValueError, match="which is reserved"):
                create_folder_structure(f"{reserved_name}/", parent_folder=temp_dir)

            capitalized = reserved_name.capitalize()
            with pytest.raises(ValueError, match="which is reserved"):
                create_folder_structure(capitalized, parent_folder=temp_dir)


@mock.patch("crewai.cli.create_crew.create_folder_structure")
@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.load_env_vars")
@mock.patch("crewai.cli.create_crew.get_provider_data")
@mock.patch("crewai.cli.create_crew.select_provider")
@mock.patch("crewai.cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_provider_param_not_shadowed_by_env_vars_loop(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_load_env_vars,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
):
    """Test that the provider parameter is not overwritten by the ENV_VARS loop variable.

    Regression test for https://github.com/crewAIInc/crewAI/issues/5270:
    The for-loop iterating over ENV_VARS used `provider` as its loop variable,
    which shadowed the `provider` function parameter. After the fix, the loop
    uses `env_provider` instead, so the original parameter value is preserved.
    """
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")

    # Simulate existing env vars that match the "openai" provider key
    mock_load_env_vars.return_value = {"OPENAI_API_KEY": "sk-existing"}
    mock_get_provider_data.return_value = {"openai": ["gpt-4"]}
    mock_select_provider.return_value = "openai"
    mock_select_model.return_value = "gpt-4"
    mock_prompt.return_value = "fake-api-key"

    # Patch click.confirm to accept the override prompt for existing provider
    with mock.patch("click.confirm", return_value=True):
        create_crew("Test Crew", provider="openai")

    # The key assertion: select_provider should still be called because
    # the provider parameter should retain its original value (not be
    # overwritten by the last ENV_VARS key from the loop).
    mock_select_provider.assert_called()


@mock.patch("crewai.cli.create_crew.create_folder_structure")
@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.load_env_vars")
@mock.patch("crewai.cli.create_crew.get_provider_data")
@mock.patch("crewai.cli.create_crew.select_provider")
@mock.patch("crewai.cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_provider_param_preserved_after_env_vars_loop_no_match(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_load_env_vars,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
):
    """Test that provider parameter is preserved when no existing env var matches.

    When no ENV_VARS key matches the loaded environment, existing_provider
    should be None and the provider parameter should still hold its original
    value (not the last key from the ENV_VARS dict).
    """
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")

    # No existing env vars — the loop iterates all of ENV_VARS without breaking
    mock_load_env_vars.return_value = {}
    mock_get_provider_data.return_value = {"anthropic": ["claude-3-opus-20240229"]}
    mock_select_provider.return_value = "anthropic"
    mock_select_model.return_value = "claude-3-opus-20240229"
    mock_prompt.return_value = "fake-api-key"

    create_crew("Test Crew", provider="anthropic")

    # Verify the function completed without errors and the provider selection
    # proceeded correctly (the provider param was not corrupted by the loop)
    mock_select_provider.assert_called()
    mock_select_model.assert_called_once_with(
        "anthropic", {"anthropic": ["claude-3-opus-20240229"]}
    )


@mock.patch("crewai.cli.create_crew.create_folder_structure")
@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.load_env_vars")
@mock.patch("crewai.cli.create_crew.get_provider_data")
@mock.patch("crewai.cli.create_crew.select_provider")
@mock.patch("crewai.cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_existing_provider_detected_without_shadowing_provider_param(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_load_env_vars,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
):
    """Test that existing_provider is correctly detected from env vars.

    Ensures the loop correctly identifies an existing provider configuration
    without corrupting the provider function parameter.
    """
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")

    # Simulate existing ANTHROPIC_API_KEY in env
    mock_load_env_vars.return_value = {"ANTHROPIC_API_KEY": "sk-ant-existing"}
    mock_get_provider_data.return_value = {"openai": ["gpt-4"]}
    mock_select_provider.return_value = "openai"
    mock_select_model.return_value = "gpt-4"
    mock_prompt.return_value = "fake-api-key"

    # Patch click.confirm — user declines to override existing provider config
    with mock.patch("click.confirm", return_value=False):
        create_crew("Test Crew", provider="openai")

    # When user declines to override, the function should return early
    # and NOT call select_provider
    mock_select_provider.assert_not_called()


@mock.patch("crewai.cli.create_crew.create_folder_structure")
@mock.patch("crewai.cli.create_crew.copy_template")
@mock.patch("crewai.cli.create_crew.load_env_vars")
@mock.patch("crewai.cli.create_crew.get_provider_data")
@mock.patch("crewai.cli.create_crew.select_provider")
@mock.patch("crewai.cli.create_crew.select_model")
@mock.patch("click.prompt")
def test_env_vars_are_uppercased_in_env_file(
    mock_prompt,
    mock_select_model,
    mock_select_provider,
    mock_get_provider_data,
    mock_load_env_vars,
    mock_copy_template,
    mock_create_folder_structure,
    tmp_path,
):
    crew_path = tmp_path / "test_crew"
    crew_path.mkdir()
    mock_create_folder_structure.return_value = (crew_path, "test_crew", "TestCrew")

    mock_load_env_vars.return_value = {}
    mock_get_provider_data.return_value = {"openai": ["gpt-4"]}
    mock_select_provider.return_value = "azure"
    mock_select_model.return_value = "azure/openai"
    mock_prompt.return_value = "fake-api-key"

    create_crew("Test Crew")

    env_file_path = crew_path / ".env"
    content = env_file_path.read_text()
    assert "MODEL=" in content

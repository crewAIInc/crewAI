from pathlib import Path
from unittest import mock

import pytest
import click
from click.testing import CliRunner

from crewai.cli.cli import (
    deploy_create,
    deploy_list,
    deploy_logs,
    deploy_push,
    deploy_remove,
    deply_status,
    flow_add_crew,
    reset_memories,
    signup,
    test,
    train,
    version,
)


from crewai.cli.cli import create
TEST_CONSTANTS = {
    "CREW_NAME": "test_crew",
    "MISTRAL_API_KEY": "mistral_api_key_123",
    "MISTRAL_MODEL": "mistral-tiny",
    "EMPTY_KEY": "",
}

@pytest.fixture
def runner():
    return CliRunner()


@mock.patch("crewai.cli.cli.train_crew")
def test_train_default_iterations(train_crew, runner):
    result = runner.invoke(train)

    train_crew.assert_called_once_with(5, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Crew for 5 iterations" in result.output


@mock.patch("crewai.cli.cli.train_crew")
def test_train_custom_iterations(train_crew, runner):
    result = runner.invoke(train, ["--n_iterations", "10"])

    train_crew.assert_called_once_with(10, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Crew for 10 iterations" in result.output


@mock.patch("crewai.cli.cli.train_crew")
def test_train_invalid_string_iterations(train_crew, runner):
    result = runner.invoke(train, ["--n_iterations", "invalid"])

    train_crew.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: train [OPTIONS]\nTry 'train --help' for help.\n\nError: Invalid value for '-n' / '--n_iterations': 'invalid' is not a valid integer.\n"
        in result.output
    )


@mock.patch("crewai.cli.reset_memories_command.ShortTermMemory")
@mock.patch("crewai.cli.reset_memories_command.EntityMemory")
@mock.patch("crewai.cli.reset_memories_command.LongTermMemory")
@mock.patch("crewai.cli.reset_memories_command.TaskOutputStorageHandler")
def test_reset_all_memories(
    MockTaskOutputStorageHandler,
    MockLongTermMemory,
    MockEntityMemory,
    MockShortTermMemory,
    runner,
):
    result = runner.invoke(reset_memories, ["--all"])
    MockShortTermMemory().reset.assert_called_once()
    MockEntityMemory().reset.assert_called_once()
    MockLongTermMemory().reset.assert_called_once()
    MockTaskOutputStorageHandler().reset.assert_called_once()

    assert result.output == "All memories have been reset.\n"


@mock.patch("crewai.cli.reset_memories_command.ShortTermMemory")
def test_reset_short_term_memories(MockShortTermMemory, runner):
    result = runner.invoke(reset_memories, ["-s"])
    MockShortTermMemory().reset.assert_called_once()
    assert result.output == "Short term memory has been reset.\n"


@mock.patch("crewai.cli.reset_memories_command.EntityMemory")
def test_reset_entity_memories(MockEntityMemory, runner):
    result = runner.invoke(reset_memories, ["-e"])
    MockEntityMemory().reset.assert_called_once()
    assert result.output == "Entity memory has been reset.\n"


@mock.patch("crewai.cli.reset_memories_command.LongTermMemory")
def test_reset_long_term_memories(MockLongTermMemory, runner):
    result = runner.invoke(reset_memories, ["-l"])
    MockLongTermMemory().reset.assert_called_once()
    assert result.output == "Long term memory has been reset.\n"


@mock.patch("crewai.cli.reset_memories_command.TaskOutputStorageHandler")
def test_reset_kickoff_outputs(MockTaskOutputStorageHandler, runner):
    result = runner.invoke(reset_memories, ["-k"])
    MockTaskOutputStorageHandler().reset.assert_called_once()
    assert result.output == "Latest Kickoff outputs stored has been reset.\n"


@mock.patch("crewai.cli.reset_memories_command.ShortTermMemory")
@mock.patch("crewai.cli.reset_memories_command.LongTermMemory")
def test_reset_multiple_memory_flags(MockShortTermMemory, MockLongTermMemory, runner):
    result = runner.invoke(
        reset_memories,
        [
            "-s",
            "-l",
        ],
    )
    MockShortTermMemory().reset.assert_called_once()
    MockLongTermMemory().reset.assert_called_once()
    assert (
        result.output
        == "Long term memory has been reset.\nShort term memory has been reset.\n"
    )


def test_reset_no_memory_flags(runner):
    result = runner.invoke(
        reset_memories,
    )
    assert (
        result.output
        == "Please specify at least one memory type to reset using the appropriate flags.\n"
    )


def test_version_flag(runner):
    result = runner.invoke(version)

    assert result.exit_code == 0
    assert "crewai version:" in result.output


def test_version_command(runner):
    result = runner.invoke(version)

    assert result.exit_code == 0
    assert "crewai version:" in result.output


def test_version_command_with_tools(runner):
    result = runner.invoke(version, ["--tools"])

    assert result.exit_code == 0
    assert "crewai version:" in result.output
    assert (
        "crewai tools version:" in result.output
        or "crewai tools not installed" in result.output
    )


@mock.patch("crewai.cli.cli.evaluate_crew")
def test_test_default_iterations(evaluate_crew, runner):
    result = runner.invoke(test)

    evaluate_crew.assert_called_once_with(3, "gpt-4o-mini")
    assert result.exit_code == 0
    assert "Testing the crew for 3 iterations with model gpt-4o-mini" in result.output


@mock.patch("crewai.cli.cli.evaluate_crew")
def test_test_custom_iterations(evaluate_crew, runner):
    result = runner.invoke(test, ["--n_iterations", "5", "--model", "gpt-4o"])

    evaluate_crew.assert_called_once_with(5, "gpt-4o")
    assert result.exit_code == 0
    assert "Testing the crew for 5 iterations with model gpt-4o" in result.output


@mock.patch("crewai.cli.cli.evaluate_crew")
def test_test_invalid_string_iterations(evaluate_crew, runner):
    result = runner.invoke(test, ["--n_iterations", "invalid"])

    evaluate_crew.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: test [OPTIONS]\nTry 'test --help' for help.\n\nError: Invalid value for '-n' / '--n_iterations': 'invalid' is not a valid integer.\n"
        in result.output
    )


@mock.patch("crewai.cli.cli.AuthenticationCommand")
def test_signup(command, runner):
    mock_auth = command.return_value
    result = runner.invoke(signup)

    assert result.exit_code == 0
    mock_auth.signup.assert_called_once()


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_create(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_create)

    assert result.exit_code == 0
    mock_deploy.create_crew.assert_called_once()


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_list(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_list)

    assert result.exit_code == 0
    mock_deploy.list_crews.assert_called_once()


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_push(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_push, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=uuid)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_push_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_push)

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=None)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_status(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deply_status, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.get_crew_status.assert_called_once_with(uuid=uuid)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_status_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deply_status)

    assert result.exit_code == 0
    mock_deploy.get_crew_status.assert_called_once_with(uuid=None)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_logs(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_logs, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.get_crew_logs.assert_called_once_with(uuid=uuid)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_logs_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_logs)

    assert result.exit_code == 0
    mock_deploy.get_crew_logs.assert_called_once_with(uuid=None)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_remove(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_remove, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.remove_crew.assert_called_once_with(uuid=uuid)


@mock.patch("crewai.cli.cli.DeployCommand")
def test_deploy_remove_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_remove)

    assert result.exit_code == 0
    mock_deploy.remove_crew.assert_called_once_with(uuid=None)


@mock.patch("crewai.cli.add_crew_to_flow.create_embedded_crew")
@mock.patch("pathlib.Path.exists", return_value=True)  # Mock the existence check
def test_flow_add_crew(mock_path_exists, mock_create_embedded_crew, runner):
    crew_name = "new_crew"
    result = runner.invoke(flow_add_crew, [crew_name])

    # Log the output for debugging
    print(result.output)

    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    assert f"Adding crew {crew_name} to the flow" in result.output

    # Verify that create_embedded_crew was called with the correct arguments
    mock_create_embedded_crew.assert_called_once()
    call_args, call_kwargs = mock_create_embedded_crew.call_args
    assert call_args[0] == crew_name
    assert "parent_folder" in call_kwargs
    assert isinstance(call_kwargs["parent_folder"], Path)


@pytest.mark.parametrize("provider,model,api_key,has_valid_keys,expected_outputs", [
    (
        "mistral",
        TEST_CONSTANTS["MISTRAL_MODEL"],
        TEST_CONSTANTS["MISTRAL_API_KEY"],
        True,
        ["API keys and model saved", f"Selected model: {TEST_CONSTANTS['MISTRAL_MODEL']}"]
    ),
    (
        "mistral",
        TEST_CONSTANTS["MISTRAL_MODEL"],
        TEST_CONSTANTS["EMPTY_KEY"],
        False,
        ["No API keys provided", f"Selected model: {TEST_CONSTANTS['MISTRAL_MODEL']}"]
    ),
    (
        "mistral",
        None,
        TEST_CONSTANTS["EMPTY_KEY"],
        False,
        ["No model selected"]
    ),
])
@mock.patch("crewai.cli.create_crew.validate_api_keys")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
@mock.patch("crewai.cli.create_crew.get_provider_data")
@mock.patch("crewai.cli.create_crew.select_model")
@mock.patch("crewai.cli.create_crew.select_provider")
@mock.patch("crewai.cli.create_crew.click.confirm")
@mock.patch("crewai.cli.create_crew.click.prompt")
def test_create_crew_scenarios(
    mock_prompt, mock_confirm, mock_select_provider, mock_select_model,
    mock_get_provider_data, mock_load_env_vars, mock_write_env_file, mock_validate_api_keys,
    runner, provider, model, api_key, has_valid_keys, expected_outputs
):
    """Test different scenarios for crew creation with provider configuration.
    
    Args:
        mock_*: Mock objects for various dependencies
        runner: Click test runner
        provider: Provider to test (e.g. "mistral")
        model: Model to select (e.g. "mistral-tiny")
        api_key: API key to provide
        has_valid_keys: Whether the API key validation should pass
        expected_output: Expected message in the output
    """
    mock_confirm.return_value = True
    mock_get_provider_data.return_value = {"mistral": [TEST_CONSTANTS["MISTRAL_MODEL"]]}
    mock_load_env_vars.return_value = {}
    mock_select_provider.return_value = provider
    mock_select_model.return_value = model
    mock_prompt.return_value = api_key
    mock_validate_api_keys.return_value = has_valid_keys

    # When model is None, simulate model selection being cancelled
    if model is None:
        mock_select_model.side_effect = click.UsageError("No model selected")

    result = runner.invoke(create, ["crew", TEST_CONSTANTS["CREW_NAME"]], input="y\n")
    
    # For model=None case, we expect error message
    if model is None:
        assert result.exit_code == 2  # UsageError exit code
        assert "No model selected" in result.output
    else:
        assert result.exit_code == 0
        for expected_output in expected_outputs:
            assert expected_output in result.output

@mock.patch("crewai.cli.create_crew.validate_api_keys")
@mock.patch("crewai.cli.create_crew.write_env_file")
@mock.patch("crewai.cli.create_crew.load_env_vars")
@mock.patch("crewai.cli.create_crew.get_provider_data")
@mock.patch("crewai.cli.create_crew.select_model")
@mock.patch("crewai.cli.create_crew.select_provider")
@mock.patch("crewai.cli.create_crew.click.confirm")
@mock.patch("crewai.cli.create_crew.click.prompt")
def test_create_crew_with_file_error(
    mock_prompt, mock_confirm, mock_select_provider, mock_select_model,
    mock_get_provider_data, mock_load_env_vars, mock_write_env_file, mock_validate_api_keys,
    runner
):
    # Mock folder override confirmation
    mock_confirm.return_value = True
    # Mock provider data
    mock_get_provider_data.return_value = {"mistral": [TEST_CONSTANTS["MISTRAL_MODEL"]]}
    # Mock empty env vars
    mock_load_env_vars.return_value = {}
    # Mock provider and model selection
    mock_select_provider.return_value = "mistral"
    mock_select_model.return_value = TEST_CONSTANTS["MISTRAL_MODEL"]
    # Mock API key input
    mock_prompt.return_value = TEST_CONSTANTS["MISTRAL_API_KEY"]
    # Mock API key validation
    mock_validate_api_keys.return_value = True
    # Mock file write error
    mock_write_env_file.side_effect = IOError("Permission denied")
    
    result = runner.invoke(create, ["crew", TEST_CONSTANTS["CREW_NAME"]], input="y\n")
    
    assert result.exit_code == 1
    assert "Error writing .env file: Permission denied" in result.output
    assert mock_write_env_file.called

def test_add_crew_to_flow_not_in_root(runner):
    # Simulate not being in the root of a flow project
    with mock.patch("pathlib.Path.exists", autospec=True) as mock_exists:
        # Mock Path.exists to return False when checking for pyproject.toml
        def exists_side_effect(self):
            if self.name == "pyproject.toml":
                return False  # Simulate that pyproject.toml does not exist
            return True  # All other paths exist

        mock_exists.side_effect = exists_side_effect

        result = runner.invoke(flow_add_crew, ["new_crew"])

        assert result.exit_code != 0
        assert "This command must be run from the root of a flow project." in str(
            result.output
        )

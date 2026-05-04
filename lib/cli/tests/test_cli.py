from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from crewai_cli.cli import (
    deploy_create,
    deploy_list,
    deploy_logs,
    deploy_push,
    deploy_remove,
    deply_status,
    flow_add_crew,
    login,
    reset_memories,
    test,
    train,
    version,
)


@pytest.fixture
def runner():
    return CliRunner()


@mock.patch("crewai_cli.cli.train_crew")
def test_train_default_iterations(train_crew, runner):
    result = runner.invoke(train)

    train_crew.assert_called_once_with(5, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Crew for 5 iterations" in result.output


@mock.patch("crewai_cli.cli.train_crew")
def test_train_custom_iterations(train_crew, runner):
    result = runner.invoke(train, ["--n_iterations", "10"])

    train_crew.assert_called_once_with(10, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Crew for 10 iterations" in result.output


@mock.patch("crewai_cli.cli.train_crew")
def test_train_invalid_string_iterations(train_crew, runner):
    result = runner.invoke(train, ["--n_iterations", "invalid"])

    train_crew.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: train [OPTIONS]\nTry 'train --help' for help.\n\nError: Invalid value for '-n' / '--n_iterations': 'invalid' is not a valid integer.\n"
        in result.output
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


@mock.patch("crewai_cli.cli.evaluate_crew")
def test_test_default_iterations(evaluate_crew, runner):
    result = runner.invoke(test)

    evaluate_crew.assert_called_once_with(3, "gpt-4o-mini", trained_agents_file=None)
    assert result.exit_code == 0
    assert "Testing the crew for 3 iterations with model gpt-4o-mini" in result.output


@mock.patch("crewai_cli.cli.evaluate_crew")
def test_test_custom_iterations(evaluate_crew, runner):
    result = runner.invoke(test, ["--n_iterations", "5", "--model", "gpt-4o"])

    evaluate_crew.assert_called_once_with(5, "gpt-4o", trained_agents_file=None)
    assert result.exit_code == 0
    assert "Testing the crew for 5 iterations with model gpt-4o" in result.output


@mock.patch("crewai_cli.cli.evaluate_crew")
def test_test_invalid_string_iterations(evaluate_crew, runner):
    result = runner.invoke(test, ["--n_iterations", "invalid"])

    evaluate_crew.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: test [OPTIONS]\nTry 'test --help' for help.\n\nError: Invalid value for '-n' / '--n_iterations': 'invalid' is not a valid integer.\n"
        in result.output
    )


@mock.patch("crewai_cli.cli.AuthenticationCommand")
def test_login(command, runner):
    mock_auth = command.return_value
    result = runner.invoke(login)

    assert result.exit_code == 0
    mock_auth.login.assert_called_once()


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_create(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_create)

    assert result.exit_code == 0
    mock_deploy.create_crew.assert_called_once()


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_list(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_list)

    assert result.exit_code == 0
    mock_deploy.list_crews.assert_called_once()


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_push(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_push, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=uuid, skip_validate=False)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_push_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_push)

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=None, skip_validate=False)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_status(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deply_status, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.get_crew_status.assert_called_once_with(uuid=uuid)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_status_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deply_status)

    assert result.exit_code == 0
    mock_deploy.get_crew_status.assert_called_once_with(uuid=None)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_logs(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_logs, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.get_crew_logs.assert_called_once_with(uuid=uuid)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_logs_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_logs)

    assert result.exit_code == 0
    mock_deploy.get_crew_logs.assert_called_once_with(uuid=None)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_remove(command, runner):
    mock_deploy = command.return_value
    uuid = "test-uuid"
    result = runner.invoke(deploy_remove, ["-u", uuid])

    assert result.exit_code == 0
    mock_deploy.remove_crew.assert_called_once_with(uuid=uuid)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_remove_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_remove)

    assert result.exit_code == 0
    mock_deploy.remove_crew.assert_called_once_with(uuid=None)


@mock.patch("crewai_cli.add_crew_to_flow.create_embedded_crew")
@mock.patch("pathlib.Path.exists", return_value=True)
def test_flow_add_crew(mock_path_exists, mock_create_embedded_crew, runner):
    crew_name = "new_crew"
    result = runner.invoke(flow_add_crew, [crew_name])

    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    assert f"Adding crew {crew_name} to the flow" in result.output

    mock_create_embedded_crew.assert_called_once()
    call_args, call_kwargs = mock_create_embedded_crew.call_args
    assert call_args[0] == crew_name
    assert "parent_folder" in call_kwargs
    assert isinstance(call_kwargs["parent_folder"], Path)


def test_add_crew_to_flow_not_in_root(runner):
    with mock.patch("pathlib.Path.exists", autospec=True) as mock_exists:
        def exists_side_effect(self):
            if self.name == "pyproject.toml":
                return False
            return True

        mock_exists.side_effect = exists_side_effect

        result = runner.invoke(flow_add_crew, ["new_crew"])

        assert result.exit_code != 0
        assert "This command must be run from the root of a flow project." in str(
            result.output
        )

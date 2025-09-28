from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from crewai.cli.cli import (
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
from crewai.crew import Crew


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


@pytest.fixture
def mock_crew():
    _mock = mock.Mock(spec=Crew, name="test_crew")
    _mock.name = "test_crew"
    return _mock


@pytest.fixture
def mock_get_crews(mock_crew):
    with mock.patch(
        "crewai.cli.reset_memories_command.get_crews", return_value=[mock_crew]
    ) as mock_get_crew:
        yield mock_get_crew


def test_reset_all_memories(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-a"])

    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="all")
        assert (
            f"[Crew ({crew.name})] Reset memories command has been completed."
            in result.output
        )
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_short_term_memories(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-s"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="short")
        assert (
            f"[Crew ({crew.name})] Short term memory has been reset." in result.output
        )
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_entity_memories(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-e"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="entity")
        assert f"[Crew ({crew.name})] Entity memory has been reset." in result.output
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_long_term_memories(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-l"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="long")
        assert f"[Crew ({crew.name})] Long term memory has been reset." in result.output
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_kickoff_outputs(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-k"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="kickoff_outputs")
        assert (
            f"[Crew ({crew.name})] Latest Kickoff outputs stored has been reset."
            in result.output
        )
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_multiple_memory_flags(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-s", "-l"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_has_calls(
            [mock.call(command_type="long"), mock.call(command_type="short")]
        )
        assert (
            f"[Crew ({crew.name})] Long term memory has been reset.\n"
            f"[Crew ({crew.name})] Short term memory has been reset.\n" in result.output
        )
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_knowledge(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["--knowledge"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="knowledge")
        assert f"[Crew ({crew.name})] Knowledge has been reset." in result.output
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_agent_knowledge(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["--agent-knowledge"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="agent_knowledge")
        assert f"[Crew ({crew.name})] Agents knowledge has been reset." in result.output
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_memory_from_many_crews(mock_get_crews, runner):
    crews = []
    for crew_id in ["id-1234", "id-5678"]:
        mock_crew = mock.Mock(spec=Crew)
        mock_crew.name = None
        mock_crew.id = crew_id
        crews.append(mock_crew)

    mock_get_crews.return_value = crews

    # Run the command
    result = runner.invoke(reset_memories, ["--knowledge"])

    call_count = 0
    for crew in crews:
        call_count += 1
        crew.reset_memories.assert_called_once_with(command_type="knowledge")
        assert f"[Crew ({crew.id})] Knowledge has been reset." in result.output

    assert call_count == 2, "reset_memories should have been called twice"


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
def test_login(command, runner):
    mock_auth = command.return_value
    result = runner.invoke(login)

    assert result.exit_code == 0
    mock_auth.login.assert_called_once()


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

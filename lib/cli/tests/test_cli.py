from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner
from crewai_cli.cli import (
    create,
    deploy_create,
    deploy_list,
    deploy_logs,
    deploy_push,
    deploy_remove,
    deply_status,
    flow_add_crew,
    flow_run,
    login,
    reset_memories,
    run,
    starter,
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

    evaluate_crew.assert_called_once_with(3, "gpt-5.4-mini", trained_agents_file=None)
    assert result.exit_code == 0
    assert "Testing the crew for 3 iterations with model gpt-5.4-mini" in result.output


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


@mock.patch("crewai_cli.cli.run_crew")
def test_run_uses_project_runner_by_default(run_crew, runner):
    result = runner.invoke(run)

    assert result.exit_code == 0
    run_crew.assert_called_once_with(
        trained_agents_file=None,
        definition=None,
        inputs=None,
    )
    assert "experimental" not in result.output.lower()


@mock.patch("crewai_cli.cli.run_crew")
def test_run_with_definition_uses_project_runner(run_crew, runner):
    result = runner.invoke(
        run,
        ["--definition", "flow.yaml", "--inputs", '{"topic":"AI"}'],
    )

    assert result.exit_code == 0
    run_crew.assert_called_once_with(
        trained_agents_file=None,
        definition="flow.yaml",
        inputs='{"topic":"AI"}',
    )


@mock.patch("crewai_cli.cli.run_crew")
def test_run_rejects_inputs_without_definition(run_crew, runner):
    result = runner.invoke(run, ["--inputs", '{"topic":"AI"}'])

    assert result.exit_code == 2
    assert "Error: --inputs requires --definition" in result.output
    run_crew.assert_not_called()


@mock.patch("crewai_cli.cli.run_crew")
def test_run_rejects_filename_with_definition(run_crew, runner):
    result = runner.invoke(run, ["--definition", "flow.yaml", "--filename", "x.pkl"])

    assert result.exit_code == 2
    assert "Error: --filename can only be used when running crews" in result.output
    run_crew.assert_not_called()


@mock.patch("crewai_cli.cli.run_crew")
def test_run_passes_filename_to_project_runner(run_crew, runner):
    result = runner.invoke(run, ["--filename", "trained.pkl"])

    assert result.exit_code == 0
    run_crew.assert_called_once_with(
        trained_agents_file="trained.pkl",
        definition=None,
        inputs=None,
    )


@mock.patch("crewai_cli.cli.TemplateCommand")
def test_starter_without_name_lists_starter_packs(template_command, runner):
    result = runner.invoke(starter)

    assert result.exit_code == 0
    template_command.return_value.list_starter_packs.assert_called_once_with()
    template_command.return_value.add_starter_pack.assert_not_called()


@mock.patch("crewai_cli.cli.TemplateCommand")
def test_starter_list_alias_lists_starter_packs(template_command, runner):
    result = runner.invoke(starter, ["list"])

    assert result.exit_code == 0
    template_command.return_value.list_starter_packs.assert_called_once_with()
    template_command.return_value.add_starter_pack.assert_not_called()


@mock.patch("crewai_cli.cli.TemplateCommand")
def test_starter_installs_named_pack(template_command, runner):
    result = runner.invoke(starter, ["analyst"])

    assert result.exit_code == 0
    template_command.return_value.add_starter_pack.assert_called_once_with(
        "analyst", None
    )
    template_command.return_value.list_starter_packs.assert_not_called()


@mock.patch("crewai_cli.cli.TemplateCommand")
def test_starter_installs_named_pack_to_target_dir(template_command, runner):
    result = runner.invoke(starter, ["analyst", "my_analyst"])

    assert result.exit_code == 0
    template_command.return_value.add_starter_pack.assert_called_once_with(
        "analyst", "my_analyst"
    )


@mock.patch("crewai_cli.cli.TemplateCommand")
def test_starter_installs_named_pack_to_output_dir(template_command, runner):
    result = runner.invoke(starter, ["analyst", "--output-dir", "my_analyst"])

    assert result.exit_code == 0
    template_command.return_value.add_starter_pack.assert_called_once_with(
        "analyst", "my_analyst"
    )


@mock.patch("crewai_cli.cli.TemplateCommand")
def test_starter_rejects_two_output_dirs(template_command, runner):
    result = runner.invoke(
        starter, ["analyst", "my_analyst", "--output-dir", "other"]
    )

    assert result.exit_code == 2
    assert "Use either TARGET_DIR or --output-dir, not both." in result.output
    template_command.return_value.add_starter_pack.assert_not_called()


@mock.patch("crewai_cli.cli.run_crew")
def test_flow_kickoff_is_deprecated_and_uses_run_path(run_crew, runner):
    result = runner.invoke(flow_run)

    assert result.exit_code == 0
    run_crew.assert_called_once_with(
        trained_agents_file=None,
        definition=None,
        inputs=None,
    )
    assert (
        "The command 'crewai flow kickoff' is deprecated. Use 'crewai run' instead."
        in result.output
    )


@mock.patch("crewai_cli.create_json_crew.create_json_crew")
def test_create_crew_in_dmn_mode_skips_provider_prompts(create_json_crew, runner):
    result = runner.invoke(create, ["crew", "DMN Crew"], env={"CREWAI_DMN": "True"})

    assert result.exit_code == 0
    create_json_crew.assert_called_once_with("DMN Crew", None, True)


@mock.patch("crewai_cli.create_flow.create_flow")
def test_create_flow_declarative_uses_declarative_scaffold(create_flow, runner):
    result = runner.invoke(create, ["flow", "My Flow", "--declarative"])

    assert result.exit_code == 0
    create_flow.assert_called_once_with("My Flow", declarative=True)


@mock.patch("crewai_cli.create_json_crew.create_json_crew")
def test_create_crew_rejects_declarative_flag(create_json_crew, runner):
    result = runner.invoke(create, ["crew", "My Crew", "--declarative"])

    assert result.exit_code == 2
    assert "--declarative can only be used with flow projects" in result.output
    create_json_crew.assert_not_called()


def test_create_requires_type_in_dmn_mode(runner):
    result = runner.invoke(create, env={"CREWAI_DMN": "True"})

    assert result.exit_code == 2
    assert "TYPE is required when CREWAI_DMN is set" in result.output


def test_create_requires_name_in_dmn_mode(runner):
    result = runner.invoke(create, ["flow"], env={"CREWAI_DMN": "True"})

    assert result.exit_code == 2
    assert "NAME is required when CREWAI_DMN is set" in result.output


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

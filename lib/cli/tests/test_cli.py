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
    evaluate_crew,
    flow_add_crew,
    flow_run,
    log_tasks_outputs,
    login,
    replay,
    reset_memories,
    run,
    test,
    traces_status,
    train,
    version,
)
from crewai_core.user_data import update_user_data


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
    result = runner.invoke(train, ["--n-iterations", "10"])

    train_crew.assert_called_once_with(10, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Crew for 10 iterations" in result.output


@mock.patch("crewai_cli.cli.train_crew")
def test_train_custom_iterations_snake_case_alias(train_crew, runner):
    result = runner.invoke(train, ["--n_iterations", "10"])

    train_crew.assert_called_once_with(10, "trained_agents_data.pkl")
    assert result.exit_code == 0
    assert "Training the Crew for 10 iterations" in result.output
    assert (
        "Warning: The flag '--n_iterations' is deprecated. Use '--n-iterations' instead."
        in result.output
    )


@mock.patch("crewai_cli.cli.train_crew")
def test_train_invalid_string_iterations(train_crew, runner):
    result = runner.invoke(train, ["--n-iterations", "invalid"])

    train_crew.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: train [OPTIONS]\nTry 'train --help' for help.\n\nError: Invalid value for '-n' / '--n-iterations': 'invalid' is not a valid integer.\n"
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
    result = runner.invoke(test, ["--n-iterations", "5", "--model", "gpt-4o"])

    evaluate_crew.assert_called_once_with(5, "gpt-4o", trained_agents_file=None)
    assert result.exit_code == 0
    assert "Testing the crew for 5 iterations with model gpt-4o" in result.output


@mock.patch("crewai_cli.cli.evaluate_crew")
def test_test_custom_iterations_snake_case_alias(evaluate_crew, runner):
    result = runner.invoke(test, ["--n_iterations", "5", "--model", "gpt-4o"])

    evaluate_crew.assert_called_once_with(5, "gpt-4o", trained_agents_file=None)
    assert result.exit_code == 0
    assert "Testing the crew for 5 iterations with model gpt-4o" in result.output
    assert (
        "Warning: The flag '--n_iterations' is deprecated. Use '--n-iterations' instead."
        in result.output
    )


@mock.patch("crewai_cli.cli.evaluate_crew")
def test_test_invalid_string_iterations(evaluate_crew, runner):
    result = runner.invoke(test, ["--n-iterations", "invalid"])

    evaluate_crew.assert_not_called()
    assert result.exit_code == 2
    assert (
        "Usage: test [OPTIONS]\nTry 'test --help' for help.\n\nError: Invalid value for '-n' / '--n-iterations': 'invalid' is not a valid integer.\n"
        in result.output
    )


def test_train_help_hides_legacy_flag_aliases(runner):
    result = runner.invoke(train, ["--help"])

    assert result.exit_code == 0
    assert "--n-iterations" in result.output
    assert "--n_iterations" not in result.output


def test_test_help_hides_legacy_flag_aliases(runner):
    result = runner.invoke(test, ["--help"])

    assert result.exit_code == 0
    assert "--n-iterations" in result.output
    assert "--n_iterations" not in result.output


def test_create_help_hides_legacy_flag_aliases(runner):
    result = runner.invoke(create, ["--help"])

    assert result.exit_code == 0
    assert "--skip-provider" in result.output
    assert "--skip_provider" not in result.output


def test_replay_help_hides_legacy_flag_aliases(runner):
    result = runner.invoke(replay, ["--help"])

    assert result.exit_code == 0
    assert "--task-id" in result.output
    assert "--task_id" not in result.output


@mock.patch("crewai_cli.create_json_crew.create_json_crew")
def test_create_skip_provider_flag(create_json_crew, runner):
    result = runner.invoke(create, ["crew", "my-crew", "--skip-provider"])

    assert result.exit_code == 0, result.output
    create_json_crew.assert_called_once_with("my-crew", None, True)
    assert "deprecated" not in result.output.lower()


@mock.patch("crewai_cli.create_json_crew.create_json_crew")
def test_create_skip_provider_snake_case_alias_warns(create_json_crew, runner):
    result = runner.invoke(create, ["crew", "my-crew", "--skip_provider"])

    assert result.exit_code == 0, result.output
    create_json_crew.assert_called_once_with("my-crew", None, True)
    assert (
        "Warning: The flag '--skip_provider' is deprecated. Use '--skip-provider' instead."
        in result.output
    )


@mock.patch("crewai_cli.cli.replay_task_command")
def test_replay_task_id_flag(replay_task_command, runner):
    result = runner.invoke(replay, ["--task-id", "task_123"])

    assert result.exit_code == 0, result.output
    replay_task_command.assert_called_once_with("task_123", trained_agents_file=None)
    assert "Replaying the crew from task task_123" in result.output


@mock.patch("crewai_cli.cli.replay_task_command")
def test_replay_task_id_snake_case_alias_warns(replay_task_command, runner):
    result = runner.invoke(replay, ["--task_id", "task_123"])

    assert result.exit_code == 0, result.output
    replay_task_command.assert_called_once_with("task_123", trained_agents_file=None)
    assert (
        "Warning: The flag '--task_id' is deprecated. Use '--task-id' instead."
        in result.output
    )


@mock.patch("crewai_cli.cli.replay_task_command")
def test_replay_without_task_id(replay_task_command, runner):
    result = runner.invoke(replay)

    assert result.exit_code == 0, result.output
    replay_task_command.assert_called_once_with(None, trained_agents_file=None)
    assert "Replaying the crew from task None" in result.output


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
def test_run_inputs_without_definition_calls_run_crew(run_crew, runner):
    # --inputs no longer requires --definition; the resolution happens in run_crew.
    result = runner.invoke(run, ["--inputs", '{"topic":"AI"}'])

    assert result.exit_code == 0
    run_crew.assert_called_once_with(
        trained_agents_file=None, definition=None, inputs='{"topic":"AI"}'
    )


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
    assert "crew, flow, tool, skill, template" in result.output


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
    mock_deploy.deploy.assert_called_once_with(uuid=uuid, skip_validate=False, force=False)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_push_force_goes_ahead_whatever_runs(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_push, ["-u", "test-uuid", "--force"])

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid="test-uuid", skip_validate=False, force=True)


@mock.patch("crewai_cli.cli.DeployCommand")
def test_deploy_push_no_uuid(command, runner):
    mock_deploy = command.return_value
    result = runner.invoke(deploy_push)

    assert result.exit_code == 0
    mock_deploy.deploy.assert_called_once_with(uuid=None, skip_validate=False, force=False)


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


def _flow_project(
    root: Path, project_name: str | None, package_dirs: list[str]
) -> Path:
    """Lay out a flow project at ``root`` with ``src/<dir>/crews`` per package dir."""
    root.mkdir(parents=True)
    name_line = f'name = "{project_name}"\n' if project_name else ""
    (root / "pyproject.toml").write_text(f'[project]\n{name_line}version = "0.1.0"\n')
    for package_dir in package_dirs:
        (root / "src" / package_dir / "crews").mkdir(parents=True)
    return root


@pytest.mark.parametrize(
    ("folder", "project_name", "package_dir"),
    [
        ("lead_qualifier", "lead_qualifier", "lead_qualifier"),
        ("use-case-lead-qualifier", "lead_qualifier", "lead_qualifier"),
        ("use-case-lead-qualifier", "lead-qualifier", "lead_qualifier"),
        ("checkout", "Lead Qualifier", "lead_qualifier"),
    ],
)
def test_flow_add_crew_finds_package_from_project_name(
    tmp_path, monkeypatch, runner, folder, project_name, package_dir
):
    root = _flow_project(tmp_path / folder, project_name, [package_dir])
    monkeypatch.chdir(root)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code == 0, result.output
    assert "Adding crew new_crew to the flow" in result.output
    crew_dir = root / "src" / package_dir / "crews" / "new_crew"
    assert (crew_dir / "new_crew.py").is_file()
    assert (crew_dir / "config" / "agents.yaml").is_file()
    assert (crew_dir / "config" / "tasks.yaml").is_file()


def test_flow_add_crew_prefers_project_name_over_other_crews_folders(
    tmp_path, monkeypatch, runner
):
    root = _flow_project(tmp_path / "proj", "alpha", ["alpha", "beta"])
    monkeypatch.chdir(root)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code == 0, result.output
    assert (root / "src" / "alpha" / "crews" / "new_crew" / "new_crew.py").is_file()
    assert not (root / "src" / "beta" / "crews" / "new_crew").exists()


def test_flow_add_crew_without_project_name_uses_single_crews_folder(
    tmp_path, monkeypatch, runner
):
    root = _flow_project(tmp_path / "proj", None, ["qualifier"])
    monkeypatch.chdir(root)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code == 0, result.output
    assert (root / "src" / "qualifier" / "crews" / "new_crew" / "new_crew.py").is_file()


def test_flow_add_crew_rejects_crews_folder_that_contradicts_project_name(
    tmp_path, monkeypatch, runner
):
    root = _flow_project(tmp_path / "proj", "lead-qualifier", ["other_pkg"])
    monkeypatch.chdir(root)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code != 0
    expected = Path("src", "lead_qualifier", "crews")
    assert (
        f"Crews folder does not exist in the current flow (expected {expected})."
        in result.output
    )
    assert list((root / "src" / "other_pkg" / "crews").iterdir()) == []
    assert not (root / "src" / "lead_qualifier").exists()


def test_flow_add_crew_without_project_name_rejects_several_crews_folders(
    tmp_path, monkeypatch, runner
):
    root = _flow_project(tmp_path / "proj", None, ["beta", "alpha"])
    monkeypatch.chdir(root)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code != 0
    alpha = Path("src", "alpha", "crews")
    beta = Path("src", "beta", "crews")
    assert (
        f"pyproject.toml has no [project].name and found 2 crews folders under "
        f"src/ ({alpha}, {beta}); expected exactly one."
    ) in result.output
    assert list((root / "src" / "alpha" / "crews").iterdir()) == []
    assert list((root / "src" / "beta" / "crews").iterdir()) == []


@pytest.mark.parametrize(
    ("project_name", "expected_message"),
    [
        (
            "alpha",
            "Crews folder does not exist in the current flow "
            f"(expected {Path('src', 'alpha', 'crews')}).",
        ),
        (
            None,
            "pyproject.toml has no [project].name and found 0 crews folders "
            "under src/ (none); expected exactly one.",
        ),
    ],
)
def test_flow_add_crew_without_crews_folder_errors(
    tmp_path, monkeypatch, runner, project_name, expected_message
):
    root = _flow_project(tmp_path / "proj", project_name, [])
    (root / "src" / "alpha").mkdir(parents=True)
    monkeypatch.chdir(root)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code != 0
    assert expected_message in result.output
    assert list((root / "src" / "alpha").iterdir()) == []


def test_add_crew_to_flow_not_in_root(tmp_path, monkeypatch, runner):
    (tmp_path / "src" / "proj" / "crews").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(flow_add_crew, ["new_crew"])

    assert result.exit_code != 0
    assert "This command must be run from the root of a flow project." in str(
        result.output
    )
    assert not (tmp_path / "src" / "proj" / "crews" / "new_crew").exists()


@mock.patch("crewai_cli.task_outputs.load_task_outputs")
def test_log_tasks_outputs_shows_result_not_blob(mock_load, runner):
    mock_load.return_value = [
        {
            "task_id": "abc",
            "expected_output": "EXP",
            "output": {"raw": "REAL", "messages": [{"role": "user"}]},
            "task_index": 0,
            "inputs": None,
            "was_replayed": 0,
            "timestamp": "t",
        }
    ]
    result = runner.invoke(log_tasks_outputs)

    assert result.exit_code == 0
    assert "Output: REAL" in result.output
    assert "messages" not in result.output


def test_traces_status_shows_disabled_when_env_false_overrides_consent(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setenv("CREWAI_STORAGE_DIR", "crewai_cli_traces_status")
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / app),
    )
    update_user_data({"trace_consent": True, "first_execution_done": True})
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")

    result = runner.invoke(traces_status)

    assert result.exit_code == 0
    assert "Enabled (user consented)" in result.output
    assert "DISABLED" in result.output

    monkeypatch.delenv("CREWAI_TRACING_ENABLED")
    unset = runner.invoke(traces_status)

    assert "not set" in unset.output
    assert "ENABLED" in unset.output
    assert "DISABLED" not in unset.output


@mock.patch("crewai_cli.task_outputs.load_task_outputs")
def test_log_tasks_outputs_plain_output(mock_load, runner):
    mock_load.return_value = [
        {
            "task_id": "abc",
            "expected_output": "EXP",
            "output": "REAL",
            "task_index": 0,
            "inputs": None,
            "was_replayed": 0,
            "timestamp": "t",
        }
    ]
    result = runner.invoke(log_tasks_outputs)

    assert result.exit_code == 0
    assert "Output: REAL" in result.output

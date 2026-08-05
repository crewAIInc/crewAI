"""Tests for unified `crewai create <resource>` scaffolding commands."""

from unittest import mock

import pytest
from click.testing import CliRunner

from crewai_cli.cli import create, crewai


@pytest.fixture
def runner():
    return CliRunner()


@mock.patch("crewai_cli.tools.main.ToolCommand")
def test_create_tool_invokes_tool_command(mock_tool_command_cls, runner):
    result = runner.invoke(create, ["tool", "my_tool"])

    assert result.exit_code == 0, result.output
    mock_tool_command_cls.return_value.create.assert_called_once_with("my_tool")
    assert "deprecated" not in result.output.lower()


@mock.patch("crewai_cli.tools.main.ToolCommand")
def test_tool_create_is_deprecated_and_still_works(mock_tool_command_cls, runner):
    result = runner.invoke(crewai, ["tool", "create", "my_tool"])

    assert result.exit_code == 0, result.output
    mock_tool_command_cls.return_value.create.assert_called_once_with("my_tool")
    assert (
        "Warning: The command 'crewai tool create' is deprecated. "
        "Use 'crewai create tool' instead."
        in result.output
    )


@mock.patch("crewai_cli.tools.main.ToolCommand")
@pytest.mark.parametrize("extra_args", [["--classic"], ["--declarative"], ["--provider", "openai"], ["--skip_provider"]])
def test_create_tool_rejects_crew_and_flow_flags(mock_tool_command_cls, runner, extra_args):
    result = runner.invoke(create, ["tool", "my_tool", *extra_args])

    assert result.exit_code == 2, result.output
    assert "Crew and flow options cannot be used with tool projects." in result.output
    mock_tool_command_cls.return_value.create.assert_not_called()


@mock.patch("crewai_cli.skills.main.SkillCommand")
def test_create_skill_invokes_skill_command(mock_skill_command_cls, runner):
    result = runner.invoke(create, ["skill", "my-skill"])

    assert result.exit_code == 0, result.output
    mock_skill_command_cls.return_value.create.assert_called_once_with(
        "my-skill", in_project=True
    )
    assert "deprecated" not in result.output.lower()


@mock.patch("crewai_cli.skills.main.SkillCommand")
def test_create_skill_no_project_flag(mock_skill_command_cls, runner):
    result = runner.invoke(create, ["skill", "my-skill", "--no-project"])

    assert result.exit_code == 0, result.output
    mock_skill_command_cls.return_value.create.assert_called_once_with(
        "my-skill", in_project=False
    )


@mock.patch("crewai_cli.skills.main.SkillCommand")
def test_skill_create_is_deprecated_and_still_works(mock_skill_command_cls, runner):
    result = runner.invoke(crewai, ["skill", "create", "my-skill"])

    assert result.exit_code == 0, result.output
    mock_skill_command_cls.return_value.create.assert_called_once_with(
        "my-skill", in_project=True
    )
    assert (
        "Warning: The command 'crewai skill create' is deprecated. "
        "Use 'crewai create skill' instead."
        in result.output
    )


@mock.patch("crewai_cli.skills.main.SkillCommand")
@pytest.mark.parametrize(
    "extra_args",
    [["--classic"], ["--declarative"], ["--provider", "openai"], ["--skip_provider"]],
)
def test_create_skill_rejects_crew_and_flow_flags(
    mock_skill_command_cls, runner, extra_args
):
    result = runner.invoke(create, ["skill", "my-skill", *extra_args])

    assert result.exit_code == 2, result.output
    assert "Crew and flow options cannot be used with skill projects." in result.output
    mock_skill_command_cls.return_value.create.assert_not_called()


@mock.patch("crewai_cli.skills.main.SkillCommand")
def test_create_crew_rejects_no_project_flag(mock_skill_command_cls, runner):
    result = runner.invoke(create, ["crew", "my-crew", "--no-project"])

    assert result.exit_code == 2, result.output
    assert "--no-project can only be used with skill projects." in result.output
    mock_skill_command_cls.return_value.create.assert_not_called()


@mock.patch("crewai_cli.remote_template.main.TemplateCommand")
def test_create_template_invokes_template_command(mock_template_command_cls, runner):
    result = runner.invoke(create, ["template", "my-template"])

    assert result.exit_code == 0, result.output
    mock_template_command_cls.return_value.add_template.assert_called_once_with(
        "my-template", None
    )
    assert "deprecated" not in result.output.lower()


@mock.patch("crewai_cli.remote_template.main.TemplateCommand")
def test_create_template_output_dir_flag(mock_template_command_cls, runner):
    result = runner.invoke(
        create, ["template", "my-template", "--output-dir", "custom_dir"]
    )

    assert result.exit_code == 0, result.output
    mock_template_command_cls.return_value.add_template.assert_called_once_with(
        "my-template", "custom_dir"
    )


@mock.patch("crewai_cli.remote_template.main.TemplateCommand")
def test_template_add_is_deprecated_and_still_works(mock_template_command_cls, runner):
    result = runner.invoke(
        crewai, ["template", "add", "my-template", "--output-dir", "custom_dir"]
    )

    assert result.exit_code == 0, result.output
    mock_template_command_cls.return_value.add_template.assert_called_once_with(
        "my-template", "custom_dir"
    )
    assert (
        "Warning: The command 'crewai template add' is deprecated. "
        "Use 'crewai create template' instead."
        in result.output
    )


@mock.patch("crewai_cli.remote_template.main.TemplateCommand")
@pytest.mark.parametrize(
    "extra_args",
    [["--classic"], ["--declarative"], ["--provider", "openai"], ["--skip_provider"]],
)
def test_create_template_rejects_crew_and_flow_flags(
    mock_template_command_cls, runner, extra_args
):
    result = runner.invoke(create, ["template", "my-template", *extra_args])

    assert result.exit_code == 2, result.output
    assert (
        "Crew and flow options cannot be used with template projects."
        in result.output
    )
    mock_template_command_cls.return_value.add_template.assert_not_called()


@mock.patch("crewai_cli.remote_template.main.TemplateCommand")
def test_create_crew_rejects_output_dir_flag(mock_template_command_cls, runner):
    result = runner.invoke(create, ["crew", "my-crew", "--output-dir", "custom_dir"])

    assert result.exit_code == 2, result.output
    assert "--output-dir can only be used with template projects." in result.output
    mock_template_command_cls.return_value.add_template.assert_not_called()

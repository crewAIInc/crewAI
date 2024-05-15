from click.testing import CliRunner
from crewai.cli.cli import version


def test_version_command():
    runner = CliRunner()
    result = runner.invoke(version)
    assert result.exit_code == 0
    assert "crewai version:" in result.output


def test_version_command_with_tools():
    runner = CliRunner()
    result = runner.invoke(version, ["--tools"])
    assert result.exit_code == 0
    assert "crewai version:" in result.output
    assert (
        "crewai tools version:" in result.output
        or "crewai tools not installed" in result.output
    )

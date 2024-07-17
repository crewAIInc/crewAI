from unittest import mock

import pytest
from click.testing import CliRunner

from crewai.cli.cli import train, version, reset_memories


@pytest.fixture
def runner():
    return CliRunner()


@mock.patch("crewai.cli.cli.train_crew")
def test_train_default_iterations(train_crew, runner):
    result = runner.invoke(train)

    train_crew.assert_called_once_with(5)
    assert result.exit_code == 0
    assert "Training the crew for 5 iterations" in result.output


@mock.patch("crewai.cli.cli.train_crew")
def test_train_custom_iterations(train_crew, runner):
    result = runner.invoke(train, ["--n_iterations", "10"])

    train_crew.assert_called_once_with(10)
    assert result.exit_code == 0
    assert "Training the crew for 10 iterations" in result.output


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

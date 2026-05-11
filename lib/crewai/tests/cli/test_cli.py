"""Tests for CLI commands that require crewai core (reset-memories).

Non-core CLI tests (train, test, version, deploy, login, flow_add_crew)
have moved to lib/cli/tests/test_cli.py.
"""

from unittest import mock

import pytest
from click.testing import CliRunner
from crewai_cli.cli import reset_memories
from crewai.crew import Crew


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_crew():
    _mock = mock.Mock(spec=Crew, name="test_crew")
    _mock.name = "test_crew"
    return _mock


@pytest.fixture
def mock_get_crews(mock_crew):
    with mock.patch(
        "crewai.utilities.reset_memories.get_crews", return_value=[mock_crew]
    ) as mock_get_crew, mock.patch(
        "crewai.utilities.reset_memories.get_flows", return_value=[]
    ):
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


def test_reset_memory(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-m"])
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="memory")
        assert (
            f"[Crew ({crew.name})] Memory has been reset." in result.output
        )
        call_count += 1

    assert call_count == 1, "reset_memories should have been called once"


def test_reset_short_flag_deprecated_maps_to_memory(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-s"])
    assert "deprecated" in result.output.lower()
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="memory")
        assert f"[Crew ({crew.name})] Memory has been reset." in result.output


def test_reset_entity_flag_deprecated_maps_to_memory(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-e"])
    assert "deprecated" in result.output.lower()
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="memory")
        assert f"[Crew ({crew.name})] Memory has been reset." in result.output


def test_reset_long_flag_deprecated_maps_to_memory(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-l"])
    assert "deprecated" in result.output.lower()
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="memory")
        assert f"[Crew ({crew.name})] Memory has been reset." in result.output


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


def test_reset_multiple_legacy_flags_collapsed_to_single_memory_reset(mock_get_crews, runner):
    result = runner.invoke(reset_memories, ["-s", "-l"])
    # Both legacy flags collapse to a single --memory reset
    assert "deprecated" in result.output.lower()
    call_count = 0
    for crew in mock_get_crews.return_value:
        crew.reset_memories.assert_called_once_with(command_type="memory")
        assert f"[Crew ({crew.name})] Memory has been reset." in result.output
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


@pytest.fixture
def mock_flow():
    _mock = mock.Mock()
    _mock.name = "TestFlow"
    _mock.memory = mock.Mock()
    _mock.memory.reset = mock.Mock()
    return _mock


@pytest.fixture
def mock_get_flows(mock_flow):
    with mock.patch(
        "crewai.utilities.reset_memories.get_flows", return_value=[mock_flow]
    ) as mock_get_flow, mock.patch(
        "crewai.utilities.reset_memories.get_crews", return_value=[]
    ):
        yield mock_get_flow


def test_reset_flow_memory(mock_get_flows, mock_flow, runner):
    result = runner.invoke(reset_memories, ["-m"])
    mock_flow.memory.reset.assert_called_once()
    assert "[Flow (TestFlow)] Memory has been reset." in result.output


def test_reset_flow_all_memories(mock_get_flows, mock_flow, runner):
    result = runner.invoke(reset_memories, ["-a"])
    mock_flow.memory.reset.assert_called_once()
    assert "[Flow (TestFlow)] Reset memories command has been completed." in result.output


def test_reset_flow_knowledge_no_effect(mock_get_flows, mock_flow, runner):
    result = runner.invoke(reset_memories, ["--knowledge"])
    mock_flow.memory.reset.assert_not_called()
    assert "[Flow (TestFlow)]" not in result.output


def test_reset_no_crew_or_flow_found(runner):
    with mock.patch(
        "crewai.utilities.reset_memories.get_crews", return_value=[]
    ), mock.patch(
        "crewai.utilities.reset_memories.get_flows", return_value=[]
    ):
        result = runner.invoke(reset_memories, ["-m"])
        assert "No crew or flow found." in result.output


def test_reset_crew_and_flow_memory(mock_crew, mock_flow, runner):
    with mock.patch(
        "crewai.utilities.reset_memories.get_crews", return_value=[mock_crew]
    ), mock.patch(
        "crewai.utilities.reset_memories.get_flows", return_value=[mock_flow]
    ):
        result = runner.invoke(reset_memories, ["-m"])
        mock_crew.reset_memories.assert_called_once_with(command_type="memory")
        mock_flow.memory.reset.assert_called_once()
        assert f"[Crew ({mock_crew.name})] Memory has been reset." in result.output
        assert "[Flow (TestFlow)] Memory has been reset." in result.output


def test_reset_flow_memory_none(runner):
    mock_flow = mock.Mock()
    mock_flow.name = "NoMemFlow"
    mock_flow.memory = None
    with mock.patch(
        "crewai.utilities.reset_memories.get_crews", return_value=[]
    ), mock.patch(
        "crewai.utilities.reset_memories.get_flows", return_value=[mock_flow]
    ):
        result = runner.invoke(reset_memories, ["-m"])
        assert "[Flow (NoMemFlow)] Memory has been reset." in result.output


def test_reset_no_memory_flags(runner):
    result = runner.invoke(
        reset_memories,
    )
    assert (
        result.output
        == "Please specify at least one memory type to reset using the appropriate flags.\n"
    )

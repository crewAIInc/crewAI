"""Test that all public API classes are properly importable."""

import importlib
from pathlib import Path
import sys

import pytest


def test_task_output_import():
    """Test that TaskOutput can be imported from crewai."""
    from crewai import TaskOutput

    assert TaskOutput is not None


def test_crew_output_import():
    """Test that CrewOutput can be imported from crewai."""
    from crewai import CrewOutput

    assert CrewOutput is not None


def test_crew_base_import():
    """Test that the legacy CrewBase import remains available from crewai."""
    from crewai import CrewBase
    from crewai.project import CrewBase as ProjectCrewBase

    assert CrewBase is ProjectCrewBase


def test_persist_is_importable_from_every_flow_path():
    """AGENTS.md scaffolded by 1.10.0-1.15.28 taught `crewai.flow.flow`."""
    from crewai.flow import persist as from_flow
    from crewai.flow.flow import persist as from_flow_flow
    from crewai.flow.persistence import persist as from_persistence

    assert from_flow_flow is from_flow is from_persistence


def test_persist_is_in_flow_flow_star_exports():
    from crewai.flow import flow

    assert "persist" in flow.__all__


# Modules whose namespace `crewai/__init__.py` extends with base_agent's globals
# to resolve forward refs. base_agent itself is the source, so it is not listed.
REBUILT_MODULES = [
    "crewai.agent.core",
    "crewai.agents.agent_executor",
    "crewai.crew",
    "crewai.flow.flow",
    "crewai.state.runtime",
    "crewai.task",
]


@pytest.mark.parametrize("module_name", REBUILT_MODULES)
def test_rebuilt_module_keeps_its_own_identity(module_name):
    importlib.import_module("crewai")  # runs the namespace rebuild

    module = sys.modules[module_name]

    assert module.__name__ == module_name
    assert module.__spec__ is not None
    assert module.__spec__.name == module_name
    assert module.__file__ is not None
    parts = tuple(module_name.split("."))
    assert Path(module.__file__).with_suffix("").parts[-len(parts) :] == parts


def test_missing_name_from_flow_flow_names_flow_flow():
    with pytest.raises(
        ImportError,
        match=r"cannot import name 'does_not_exist' from 'crewai\.flow\.flow'",
    ):
        from crewai.flow.flow import does_not_exist  # noqa: F401


def test_missing_name_from_crew_names_crew():
    with pytest.raises(
        ImportError, match=r"cannot import name 'does_not_exist' from 'crewai\.crew'"
    ):
        from crewai.crew import does_not_exist  # noqa: F401


def test_namespace_rebuild_still_resolves_forward_refs():
    """Fails, where the RuntimeState tests that skip on a failed rebuild would not."""
    from crewai import Agent, Crew, RuntimeState, Task

    assert RuntimeState is not None
    agent = Agent(role="r", goal="g", backstory="b", llm="gpt-4o-mini")
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="d", expected_output="e", agent=agent)],
    )

    restored = RuntimeState.model_validate_json(
        RuntimeState(root=[crew]).model_dump_json(),
        context={"from_checkpoint": True},
    )

    assert restored.root[0].agents[0].role == "r"
    assert restored.root[0].tasks[0].description == "d"

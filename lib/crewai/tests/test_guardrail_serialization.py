"""Tests for JSON serialization of guardrail fields on Task, Agent, and LiteAgent.

Guardrails accept either string descriptions or callables. Callables cannot be
JSON-serialized, so the checkpoint path must drop them rather than raise.
"""

import pytest

from crewai import Agent, Task
from crewai.lite_agent import LiteAgent
from crewai.utilities.guardrail import (
    serialize_guardrail_for_json,
    serialize_guardrails_for_json,
)


def _example_guardrail(output):
    return True, output


def test_serialize_guardrail_preserves_string() -> None:
    assert serialize_guardrail_for_json("validate output") == "validate output"


def test_serialize_guardrail_returns_none_for_none() -> None:
    assert serialize_guardrail_for_json(None) is None


def test_serialize_guardrail_drops_callable_with_warning() -> None:
    with pytest.warns(UserWarning, match="cannot be JSON-serialized"):
        assert serialize_guardrail_for_json(_example_guardrail) is None


def test_serialize_guardrails_drops_callables_from_list() -> None:
    with pytest.warns(UserWarning):
        result = serialize_guardrails_for_json(["check size", _example_guardrail])
    assert result == ["check size"]


def test_serialize_guardrails_all_callables_returns_empty_list() -> None:
    with pytest.warns(UserWarning):
        result = serialize_guardrails_for_json([_example_guardrail, _example_guardrail])
    assert result == []


def test_serialize_guardrails_handles_single_string() -> None:
    assert serialize_guardrails_for_json("only check this") == "only check this"


def test_serialize_guardrails_handles_single_callable() -> None:
    with pytest.warns(UserWarning):
        assert serialize_guardrails_for_json(_example_guardrail) is None


def test_task_model_dump_json_with_string_guardrail() -> None:
    agent = Agent(role="r", goal="g", backstory="b")
    task = Task(
        description="Do the thing",
        expected_output="A thing",
        agent=agent,
        guardrail="output must be non-empty",
    )
    dumped = task.model_dump(mode="json")
    assert dumped["guardrail"] == "output must be non-empty"


def test_task_model_dump_json_with_callable_guardrail_does_not_raise() -> None:
    agent = Agent(role="r", goal="g", backstory="b")
    task = Task(
        description="Do the thing",
        expected_output="A thing",
        agent=agent,
        guardrail=_example_guardrail,
    )
    with pytest.warns(UserWarning, match="cannot be JSON-serialized"):
        dumped = task.model_dump(mode="json")
    assert dumped["guardrail"] is None


def test_task_model_dump_json_with_callable_guardrails_list() -> None:
    agent = Agent(role="r", goal="g", backstory="b")
    task = Task(
        description="Do the thing",
        expected_output="A thing",
        agent=agent,
        guardrails=[_example_guardrail, "also check this"],
    )
    with pytest.warns(UserWarning):
        dumped = task.model_dump(mode="json")
    assert dumped["guardrails"] == ["also check this"]


def test_task_guardrails_round_trip_through_model_validate() -> None:
    """Serialized guardrails must round-trip — None entries would fail validation."""
    agent = Agent(role="r", goal="g", backstory="b")
    task = Task(
        description="Do the thing",
        expected_output="A thing",
        agent=agent,
        guardrails=[_example_guardrail, "also check this"],
    )
    with pytest.warns(UserWarning):
        dumped = task.model_dump(mode="json", exclude={"id"})
    if isinstance(dumped.get("agent"), dict):
        dumped["agent"].pop("id", None)
    Task.model_validate(dumped)


def test_agent_model_dump_json_with_callable_guardrail() -> None:
    agent = Agent(
        role="r",
        goal="g",
        backstory="b",
        guardrail=_example_guardrail,
    )
    with pytest.warns(UserWarning, match="cannot be JSON-serialized"):
        dumped = agent.model_dump(mode="json")
    assert dumped["guardrail"] is None


def test_lite_agent_model_dump_json_with_callable_guardrail() -> None:
    agent = LiteAgent(
        role="r",
        goal="g",
        backstory="b",
        guardrail=_example_guardrail,
    )
    with pytest.warns(UserWarning, match="cannot be JSON-serialized"):
        dumped = agent.model_dump(mode="json")
    assert dumped["guardrail"] is None

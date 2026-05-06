"""Tests for Flow.list_tools()."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.flow.flow import Flow, start
from crewai.task import Task
from crewai.tools import BaseTool


class _Args(BaseModel):
    pass


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name: str = tool_name
        description: str = "test"
        args_schema: type = _Args

        def _run(self, **_: Any) -> str:
            return ""

    return _T()


def _crew(role: str = "writer", tool_name: str = "search") -> Crew:
    agent = Agent(role=role, goal="g", backstory="b", tools=[_tool(tool_name)])
    task = Task(description="d", expected_output="e", agent=agent)
    return Crew(agents=[agent], tasks=[task])


def test_empty_flow_returns_empty_dict():
    class EmptyFlow(Flow):
        @start()
        def kickoff(self):
            return None

    assert EmptyFlow().list_tools() == {}


def test_crew_attribute_keyed_by_attribute_name():
    class SingleCrewFlow(Flow):
        def __init__(self):
            super().__init__()
            self.poem_crew = _crew()

        @start()
        def kickoff(self):
            return self.poem_crew.kickoff()

    assert SingleCrewFlow().list_tools() == {"poem_crew": {"writer": ["search"]}}


def test_list_of_crews_keyed_with_index_suffix():
    class ListFlow(Flow):
        def __init__(self):
            super().__init__()
            self.research_crews = [_crew("a", "t1"), _crew("b", "t2")]

        @start()
        def kickoff(self):
            return None

    tools = ListFlow().list_tools()
    assert tools == {
        "research_crews[0]": {"a": ["t1"]},
        "research_crews[1]": {"b": ["t2"]},
    }


def test_tuple_of_crews_supported():
    class TupleFlow(Flow):
        def __init__(self):
            super().__init__()
            self.crews_tuple = (_crew("a", "t1"),)

        @start()
        def kickoff(self):
            return None

    assert TupleFlow().list_tools() == {"crews_tuple[0]": {"a": ["t1"]}}


def test_underscore_prefixed_attributes_ignored():
    class HiddenFlow(Flow):
        def __init__(self):
            super().__init__()
            self._private_crew = _crew("a", "t1")
            self.public_crew = _crew("b", "t2")

        @start()
        def kickoff(self):
            return None

    tools = HiddenFlow().list_tools()
    assert "_private_crew" not in tools
    assert tools == {"public_crew": {"b": ["t2"]}}


def test_non_crew_attributes_skipped():
    class MixedFlow(Flow):
        def __init__(self):
            super().__init__()
            self.label = "some-string"
            self.config = {"k": "v"}
            self.poem_crew = _crew()

        @start()
        def kickoff(self):
            return None

    assert MixedFlow().list_tools() == {"poem_crew": {"writer": ["search"]}}


def test_list_with_non_crew_items_filtered():
    class PartialFlow(Flow):
        def __init__(self):
            super().__init__()
            self.things = [_crew("a", "t1"), "not a crew", 42]

        @start()
        def kickoff(self):
            return None

    assert PartialFlow().list_tools() == {"things[0]": {"a": ["t1"]}}

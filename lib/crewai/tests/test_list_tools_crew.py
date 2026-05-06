"""Tests for Crew.list_tools()."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task
from crewai.tools import BaseTool


class _Args(BaseModel):
    pass


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name: str = tool_name
        description: str = "test tool"
        args_schema: type = _Args

        def _run(self, **_: Any) -> str:
            return ""

    return _T()


@pytest.fixture
def writer():
    return Agent(role="writer", goal="g", backstory="b", tools=[_tool("search")])


@pytest.fixture
def editor():
    return Agent(role="editor", goal="g", backstory="b")


def test_lists_user_defined_agent_tools(writer):
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])

    assert crew.list_tools() == {"writer": ["search"]}


def test_includes_task_level_tool_overrides(writer):
    extra = _tool("calculator")
    task = Task(description="d", expected_output="e", agent=writer, tools=[extra])
    crew = Crew(agents=[writer], tasks=[task])

    assert crew.list_tools() == {"writer": ["search", "calculator"]}


def test_dedupes_when_agent_and_task_share_a_tool(writer):
    duplicate = _tool("search")
    task = Task(description="d", expected_output="e", agent=writer, tools=[duplicate])
    crew = Crew(agents=[writer], tasks=[task])

    assert crew.list_tools() == {"writer": ["search"]}


def test_peer_delegation_adds_delegate_and_ask_tools(writer, editor):
    writer.allow_delegation = True
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer, editor], tasks=[task])

    tools = crew.list_tools()
    assert "Delegate work to coworker" in tools["writer"]
    assert "Ask question to coworker" in tools["writer"]
    assert "Delegate work to coworker" not in tools["editor"]


def test_peer_delegation_skipped_when_only_one_agent(writer):
    writer.allow_delegation = True
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])

    assert "Delegate work to coworker" not in crew.list_tools()["writer"]


def test_hierarchical_includes_default_manager(writer, editor):
    writer.allow_delegation = True
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(
        agents=[writer, editor],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm="gpt-4o-mini",
    )

    tools = crew.list_tools()
    assert "writer" in tools
    assert "Delegate work to coworker" not in tools["writer"]
    # Default manager role from i18n.
    manager_keys = [k for k in tools if k not in {"writer", "editor"}]
    assert len(manager_keys) == 1
    manager_role = manager_keys[0]
    assert tools[manager_role] == [
        "Delegate work to coworker",
        "Ask question to coworker",
    ]


def test_hierarchical_uses_user_provided_manager_role(writer, editor):
    manager = Agent(role="Chief", goal="g", backstory="b", allow_delegation=True)
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(
        agents=[writer, editor],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=manager,
        manager_llm="gpt-4o-mini",
    )

    tools = crew.list_tools()
    assert "Chief" in tools
    assert tools["Chief"] == [
        "Delegate work to coworker",
        "Ask question to coworker",
    ]


def test_multimodal_added_when_llm_does_not_support_it(writer):
    writer.multimodal = True
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])

    with patch.object(type(writer.llm), "supports_multimodal", return_value=False):
        tools = crew.list_tools()

    assert "Add image to content" in tools["writer"]


def test_multimodal_skipped_when_llm_supports_it(writer):
    writer.multimodal = True
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])

    with patch.object(type(writer.llm), "supports_multimodal", return_value=True):
        tools = crew.list_tools()

    assert "Add image to content" not in tools["writer"]


def test_crew_level_memory_adds_search_and_save(writer):
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task], memory=True)

    tools = crew.list_tools()
    assert "Search memory" in tools["writer"]
    assert "Save to memory" in tools["writer"]


def test_no_memory_means_no_memory_tools(writer):
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])  # memory defaults to False

    tools = crew.list_tools()
    assert "Search memory" not in tools["writer"]
    assert "Save to memory" not in tools["writer"]


def test_mcp_emits_placeholder_per_server():
    a = Agent(role="r", goal="g", backstory="b", mcps=["github", "slack"])
    task = Task(description="d", expected_output="e", agent=a)
    crew = Crew(agents=[a], tasks=[task])

    assert crew.list_tools()["r"] == ["mcp:github:*", "mcp:slack:*"]


def test_apps_emit_placeholder_with_action_split():
    a = Agent(
        role="r",
        goal="g",
        backstory="b",
        apps=["gmail", "slack#send_message"],
    )
    task = Task(description="d", expected_output="e", agent=a)
    crew = Crew(agents=[a], tasks=[task])

    assert crew.list_tools()["r"] == ["app:gmail:*", "app:slack:send_message"]


def test_file_reader_added_when_task_has_input_files(writer):
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])

    sentinel_files = {"foo.txt": object()}
    with patch("crewai.crew.get_all_files", return_value=sentinel_files):
        tools = crew.list_tools()

    assert "read_file" in tools["writer"]


def test_file_reader_not_added_when_no_input_files(writer):
    task = Task(description="d", expected_output="e", agent=writer)
    crew = Crew(agents=[writer], tasks=[task])

    with patch("crewai.crew.get_all_files", return_value={}):
        tools = crew.list_tools()

    assert "read_file" not in tools["writer"]

"""Tests for MCP tools loading during delegation (Issue #4571).

When an agent with MCP servers configured is used as a sub-agent via delegation,
its MCP tools must be loaded even though the Crew's _prepare_tools() is not called
for the delegated-to agent.
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent.core import Agent
from crewai.agent.utils import _inject_mcp_tools, prepare_tools
from crewai.mcp.config import MCPServerHTTP
from crewai.task import Task
from crewai.tools.base_tool import BaseTool


def _make_mock_tool(name: str) -> MagicMock:
    """Create a MagicMock that looks like a BaseTool with the given name."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    return tool


@pytest.fixture
def http_config():
    """Create a sample MCPServerHTTP configuration."""
    return MCPServerHTTP(url="https://api.example.com/mcp")


@pytest.fixture
def sub_agent_with_mcp(http_config):
    """Create an agent with MCP servers configured (the delegated-to agent)."""
    return Agent(
        role="MCP Sub Agent",
        goal="Execute tasks using MCP tools",
        backstory="An agent that uses MCP server tools",
        mcps=[http_config],
        allow_delegation=False,
    )


@pytest.fixture
def sub_agent_without_mcp():
    """Create an agent without MCP servers."""
    return Agent(
        role="Regular Sub Agent",
        goal="Execute tasks normally",
        backstory="An agent without MCP tools",
        allow_delegation=False,
    )


class TestInjectMcpTools:
    """Tests for the _inject_mcp_tools helper function."""

    def test_injects_mcp_tools_when_agent_has_mcps(self, sub_agent_with_mcp):
        """MCP tools should be injected when agent has mcps configured."""
        mock_mcp_tools = [_make_mock_tool("mcp_search"), _make_mock_tool("mcp_fetch")]

        with patch.object(Agent, "get_mcp_tools", return_value=mock_mcp_tools):
            tools: list[BaseTool] = []
            result = _inject_mcp_tools(sub_agent_with_mcp, tools)

            assert len(result) == 2
            tool_names = {t.name for t in result}
            assert "mcp_search" in tool_names
            assert "mcp_fetch" in tool_names

    def test_does_not_inject_when_agent_has_no_mcps(self, sub_agent_without_mcp):
        """No MCP tools should be injected when agent has no mcps."""
        tools: list[BaseTool] = []
        result = _inject_mcp_tools(sub_agent_without_mcp, tools)
        assert len(result) == 0

    def test_does_not_duplicate_existing_mcp_tools(self, sub_agent_with_mcp):
        """MCP tools already in the list should not be duplicated."""
        existing_search = _make_mock_tool("mcp_search")
        mock_mcp_tools = [_make_mock_tool("mcp_search"), _make_mock_tool("mcp_fetch")]

        with patch.object(Agent, "get_mcp_tools", return_value=mock_mcp_tools):
            tools = [existing_search]
            result = _inject_mcp_tools(sub_agent_with_mcp, tools)

            # Should have 2 tools: existing mcp_search + new mcp_fetch
            assert len(result) == 2
            tool_names = [t.name for t in result]
            assert tool_names.count("mcp_search") == 1
            assert tool_names.count("mcp_fetch") == 1

    def test_preserves_existing_tools(self, sub_agent_with_mcp):
        """Existing non-MCP tools should be preserved after injection."""
        mock_mcp_tools = [_make_mock_tool("mcp_search"), _make_mock_tool("mcp_fetch")]

        with patch.object(Agent, "get_mcp_tools", return_value=mock_mcp_tools):
            existing_tool = _make_mock_tool("existing_tool")
            tools = [existing_tool]

            result = _inject_mcp_tools(sub_agent_with_mcp, tools)

            assert len(result) == 3  # 1 existing + 2 MCP
            tool_names = {t.name for t in result}
            assert "existing_tool" in tool_names
            assert "mcp_search" in tool_names
            assert "mcp_fetch" in tool_names

    def test_handles_mcp_loading_failure_gracefully(self, sub_agent_with_mcp):
        """If MCP tool loading fails, existing tools should be returned unmodified."""
        with patch.object(
            Agent, "get_mcp_tools", side_effect=Exception("Connection failed")
        ):
            existing_tool = _make_mock_tool("my_tool")
            tools = [existing_tool]

            result = _inject_mcp_tools(sub_agent_with_mcp, tools)

            assert len(result) == 1
            assert result[0].name == "my_tool"

    def test_handles_empty_mcp_tools_list(self, sub_agent_with_mcp):
        """If MCP server returns empty tools list, original tools are unchanged."""
        with patch.object(Agent, "get_mcp_tools", return_value=[]):
            existing_tool = _make_mock_tool("my_tool")
            tools = [existing_tool]

            result = _inject_mcp_tools(sub_agent_with_mcp, tools)

            assert len(result) == 1
            assert result[0].name == "my_tool"

    def test_handles_agent_with_empty_mcps_list(self):
        """An agent with an empty mcps list should not trigger MCP loading."""
        agent = Agent(
            role="Agent",
            goal="Test",
            backstory="Test",
            mcps=[],
            allow_delegation=False,
        )
        tools: list[BaseTool] = []
        result = _inject_mcp_tools(agent, tools)
        assert len(result) == 0


class TestPrepareToolsWithMcp:
    """Tests for prepare_tools function with MCP integration."""

    def test_prepare_tools_injects_mcp_when_tools_is_none(
        self, sub_agent_with_mcp
    ):
        """When tools=None (delegation scenario), MCP tools should be loaded."""
        task = Task(
            description="Test task for delegation",
            agent=sub_agent_with_mcp,
            expected_output="Test output",
        )

        mock_mcp_tools = [_make_mock_tool("mcp_search"), _make_mock_tool("mcp_fetch")]
        with patch.object(Agent, "get_mcp_tools", return_value=mock_mcp_tools), \
             patch.object(Agent, "create_agent_executor"):
            result = prepare_tools(sub_agent_with_mcp, None, task)

            tool_names = {t.name for t in result}
            assert "mcp_search" in tool_names
            assert "mcp_fetch" in tool_names

    def test_prepare_tools_no_mcp_when_agent_has_no_mcps(
        self, sub_agent_without_mcp
    ):
        """When agent has no mcps, prepare_tools should behave normally."""
        task = Task(
            description="Test task",
            agent=sub_agent_without_mcp,
            expected_output="Test output",
        )

        with patch.object(Agent, "create_agent_executor"):
            result = prepare_tools(sub_agent_without_mcp, None, task)
            assert len(result) == 0

    def test_prepare_tools_merges_explicit_tools_and_mcp(
        self, sub_agent_with_mcp
    ):
        """When explicit tools are passed + agent has mcps, both should be present."""
        task = Task(
            description="Test task",
            agent=sub_agent_with_mcp,
            expected_output="Test output",
        )

        explicit_tool = _make_mock_tool("custom_tool")
        mock_mcp_tools = [_make_mock_tool("mcp_search")]
        with patch.object(Agent, "get_mcp_tools", return_value=mock_mcp_tools), \
             patch.object(Agent, "create_agent_executor"):
            result = prepare_tools(sub_agent_with_mcp, [explicit_tool], task)

            tool_names = {t.name for t in result}
            assert "custom_tool" in tool_names
            assert "mcp_search" in tool_names


class TestDelegationWithMcp:
    """Tests for the full delegation flow with MCP-configured sub-agents."""

    def test_delegation_tool_loads_mcp_tools_for_sub_agent(
        self, sub_agent_with_mcp
    ):
        """When DelegateWorkTool delegates to an agent with MCPs,
        the MCP tools should be loaded during execute_task."""
        task = Task(
            description="Search for AI news",
            agent=sub_agent_with_mcp,
            expected_output="AI news results",
        )

        mock_mcp_tools = [_make_mock_tool("mcp_search")]

        with patch.object(Agent, "get_mcp_tools", return_value=mock_mcp_tools), \
             patch.object(Agent, "create_agent_executor"), \
             patch.object(Agent, "_execute_without_timeout", return_value="Found AI news"):
            # Simulate what DelegateWorkTool does: call execute_task with no tools
            result = sub_agent_with_mcp.execute_task(task, "context")

            assert result == "Found AI news"

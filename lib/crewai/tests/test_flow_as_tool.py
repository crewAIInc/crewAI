"""Tests for Flow-as-tool functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

from crewai.flow.flow import Flow, start
from crewai.tools.flow_tool import FlowTool, create_flow_tools


# ---------------------------------------------------------------------------
# Test Flow classes
# ---------------------------------------------------------------------------


class SimpleFlow(Flow):
    """A simple flow that greets the user."""

    @start()
    def greet(self) -> str:
        return "Hello from SimpleFlow!"


class MathFlow(Flow):
    """Performs basic math operations."""

    @start()
    def compute(self) -> str:
        return "42"


class NoDocFlow(Flow):
    @start()
    def run_it(self) -> str:
        return "no doc"


# ---------------------------------------------------------------------------
# FlowTool unit tests
# ---------------------------------------------------------------------------


class TestFlowTool:
    def test_wrap_simple_flow(self) -> None:
        tool = FlowTool(
            name="simple_flow",
            description="A simple flow that greets the user.",
            flow_class=SimpleFlow,
        )
        assert tool.name == "simple_flow"
        assert "greets the user" in tool.description

    def test_run_invokes_kickoff(self) -> None:
        mock_flow = MagicMock()
        mock_flow.return_value = mock_flow  # __init__ returns self
        mock_flow.kickoff.return_value = "mocked result"

        tool = FlowTool(
            name="test_flow",
            description="test",
            flow_class=mock_flow,
        )
        result = tool._run(inputs="{}")
        assert result == "mocked result"
        mock_flow.kickoff.assert_called_once()

    def test_run_with_json_inputs(self) -> None:
        mock_flow = MagicMock()
        mock_flow.return_value = mock_flow
        mock_flow.kickoff.return_value = "result with inputs"

        tool = FlowTool(
            name="test_flow",
            description="test",
            flow_class=mock_flow,
        )
        result = tool._run(inputs='{"key": "value"}')
        assert result == "result with inputs"
        mock_flow.kickoff.assert_called_once_with(inputs={"key": "value"})

    def test_run_with_invalid_json_defaults_to_empty(self) -> None:
        mock_flow = MagicMock()
        mock_flow.return_value = mock_flow
        mock_flow.kickoff.return_value = "ok"

        tool = FlowTool(
            name="test_flow",
            description="test",
            flow_class=mock_flow,
        )
        result = tool._run(inputs="not valid json")
        assert result == "ok"
        mock_flow.kickoff.assert_called_once_with(inputs=None)

    def test_run_returns_string(self) -> None:
        mock_flow = MagicMock()
        mock_flow.return_value = mock_flow
        mock_flow.kickoff.return_value = 42

        tool = FlowTool(
            name="test_flow",
            description="test",
            flow_class=mock_flow,
        )
        result = tool._run()
        assert result == "42"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# create_flow_tools tests
# ---------------------------------------------------------------------------


class TestCreateFlowTools:
    def test_creates_tools_from_flow_classes(self) -> None:
        tools = create_flow_tools([SimpleFlow, MathFlow])
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "simple_flow" in names
        assert "math_flow" in names

    def test_description_from_docstring(self) -> None:
        tools = create_flow_tools([SimpleFlow])
        assert len(tools) == 1
        assert "greets the user" in tools[0].description

    def test_description_fallback_when_no_docstring(self) -> None:
        tools = create_flow_tools([NoDocFlow])
        assert len(tools) == 1
        assert "NoDocFlow" in tools[0].description

    def test_empty_list_returns_empty(self) -> None:
        assert create_flow_tools([]) == []

    def test_none_returns_empty(self) -> None:
        assert create_flow_tools(None) == []

    def test_tools_are_base_tool_instances(self) -> None:
        from crewai.tools.base_tool import BaseTool

        tools = create_flow_tools([SimpleFlow])
        for tool in tools:
            assert isinstance(tool, BaseTool)


# ---------------------------------------------------------------------------
# Agent integration tests
# ---------------------------------------------------------------------------


class TestAgentFlowIntegration:
    def test_agent_with_flows_has_flow_tools(self) -> None:
        from crewai.agent.core import Agent

        agent = Agent(
            role="Test Agent",
            goal="Test flows",
            backstory="I test things",
            flows=[SimpleFlow, MathFlow],
        )
        tool_names = {t.name for t in (agent.tools or [])}
        assert "simple_flow" in tool_names
        assert "math_flow" in tool_names

    def test_agent_without_flows_no_extra_tools(self) -> None:
        from crewai.agent.core import Agent

        agent = Agent(
            role="Test Agent",
            goal="Test",
            backstory="I test things",
        )
        # Should not have any flow tools
        flow_tool_names = {
            t.name for t in (agent.tools or []) if isinstance(t, FlowTool)
        }
        assert len(flow_tool_names) == 0

    def test_flow_tool_executes_real_flow(self) -> None:
        """Test that a FlowTool actually runs the Flow's kickoff."""
        tools = create_flow_tools([SimpleFlow])
        tool = tools[0]
        result = tool.run(inputs="{}")
        assert "Hello from SimpleFlow" in result

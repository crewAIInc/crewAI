"""Test Agent multimodal kickoff functionality."""

from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent
from crewai.lite_agent import LiteAgent
from crewai.lite_agent_output import LiteAgentOutput
from crewai.tools.agent_tools.add_image_tool import AddImageTool
from crewai.tools.base_tool import BaseTool


@pytest.fixture
def mock_lite_agent():
    """Fixture to mock LiteAgent to avoid LLM calls."""
    with patch("crewai.agent.core.LiteAgent") as mock_lite_agent_class:
        mock_instance = MagicMock(spec=LiteAgent)
        mock_output = LiteAgentOutput(
            raw="Test output",
            pydantic=None,
            agent_role="test role",
            usage_metrics=None,
            messages=[],
        )
        mock_instance.kickoff.return_value = mock_output
        mock_instance.kickoff_async.return_value = mock_output
        mock_lite_agent_class.return_value = mock_instance
        yield mock_lite_agent_class


def test_agent_kickoff_with_multimodal_true_adds_image_tool(mock_lite_agent):
    """Test that when multimodal=True, AddImageTool is added to the tools."""
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        multimodal=True,
    )

    agent.kickoff("Test message")

    mock_lite_agent.assert_called_once()
    call_kwargs = mock_lite_agent.call_args[1]
    tools = call_kwargs["tools"]

    assert any(isinstance(tool, AddImageTool) for tool in tools)


def test_agent_kickoff_with_multimodal_false_does_not_add_image_tool(mock_lite_agent):
    """Test that when multimodal=False, AddImageTool is not added to the tools."""
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        multimodal=False,
    )

    agent.kickoff("Test message")

    mock_lite_agent.assert_called_once()
    call_kwargs = mock_lite_agent.call_args[1]
    tools = call_kwargs["tools"]

    assert not any(isinstance(tool, AddImageTool) for tool in tools)


def test_agent_kickoff_does_not_mutate_self_tools(mock_lite_agent):
    """Test that calling kickoff does not mutate self.tools."""

    class DummyTool(BaseTool):
        name: str = "dummy_tool"
        description: str = "A dummy tool"

        def _run(self, **kwargs):
            return "dummy result"

    dummy_tool = DummyTool()
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[dummy_tool],
        multimodal=True,
    )

    original_tools_count = len(agent.tools)
    original_tools = list(agent.tools)

    agent.kickoff("Test message")

    assert len(agent.tools) == original_tools_count
    assert agent.tools == original_tools


def test_agent_kickoff_multiple_calls_does_not_duplicate_tools(mock_lite_agent):
    """Test that calling kickoff multiple times does not duplicate tools."""
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        multimodal=True,
    )

    agent.kickoff("Test message 1")
    first_call_tools = mock_lite_agent.call_args[1]["tools"]
    first_call_image_tools = [
        tool for tool in first_call_tools if isinstance(tool, AddImageTool)
    ]

    agent.kickoff("Test message 2")
    second_call_tools = mock_lite_agent.call_args[1]["tools"]
    second_call_image_tools = [
        tool for tool in second_call_tools if isinstance(tool, AddImageTool)
    ]

    assert len(first_call_image_tools) == 1
    assert len(second_call_image_tools) == 1


def test_agent_kickoff_async_with_multimodal_true_adds_image_tool(mock_lite_agent):
    """Test that when multimodal=True, AddImageTool is added in kickoff_async."""
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        multimodal=True,
    )

    import asyncio

    asyncio.run(agent.kickoff_async("Test message"))

    mock_lite_agent.assert_called_once()
    call_kwargs = mock_lite_agent.call_args[1]
    tools = call_kwargs["tools"]

    assert any(isinstance(tool, AddImageTool) for tool in tools)


def test_agent_kickoff_async_with_multimodal_false_does_not_add_image_tool(
    mock_lite_agent,
):
    """Test that when multimodal=False, AddImageTool is not added in kickoff_async."""
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        multimodal=False,
    )

    import asyncio

    asyncio.run(agent.kickoff_async("Test message"))

    mock_lite_agent.assert_called_once()
    call_kwargs = mock_lite_agent.call_args[1]
    tools = call_kwargs["tools"]

    assert not any(isinstance(tool, AddImageTool) for tool in tools)


def test_agent_kickoff_async_does_not_mutate_self_tools(mock_lite_agent):
    """Test that calling kickoff_async does not mutate self.tools."""

    class DummyTool(BaseTool):
        name: str = "dummy_tool"
        description: str = "A dummy tool"

        def _run(self, **kwargs):
            return "dummy result"

    dummy_tool = DummyTool()
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[dummy_tool],
        multimodal=True,
    )

    original_tools_count = len(agent.tools)
    original_tools = list(agent.tools)

    import asyncio

    asyncio.run(agent.kickoff_async("Test message"))

    assert len(agent.tools) == original_tools_count
    assert agent.tools == original_tools


def test_agent_kickoff_with_existing_tools_and_multimodal(mock_lite_agent):
    """Test that multimodal tools are added alongside existing tools."""

    class DummyTool(BaseTool):
        name: str = "dummy_tool"
        description: str = "A dummy tool"

        def _run(self, **kwargs):
            return "dummy result"

    dummy_tool = DummyTool()
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[dummy_tool],
        multimodal=True,
    )

    agent.kickoff("Test message")

    mock_lite_agent.assert_called_once()
    call_kwargs = mock_lite_agent.call_args[1]
    tools = call_kwargs["tools"]

    assert any(isinstance(tool, DummyTool) for tool in tools)
    assert any(isinstance(tool, AddImageTool) for tool in tools)
    assert len(tools) == 2


def test_agent_kickoff_deduplicates_tools_by_name(mock_lite_agent):
    """Test that tools with the same name are deduplicated."""

    class DummyTool(BaseTool):
        name: str = "dummy_tool"
        description: str = "A dummy tool"

        def _run(self, **kwargs):
            return "dummy result"

    dummy_tool1 = DummyTool()
    dummy_tool2 = DummyTool()

    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        tools=[dummy_tool1, dummy_tool2],
        multimodal=False,
    )

    agent.kickoff("Test message")

    mock_lite_agent.assert_called_once()
    call_kwargs = mock_lite_agent.call_args[1]
    tools = call_kwargs["tools"]

    dummy_tools = [tool for tool in tools if isinstance(tool, DummyTool)]
    assert len(dummy_tools) == 1


def test_agent_kickoff_async_includes_platform_and_mcp_tools(mock_lite_agent):
    """Test that kickoff_async includes platform and MCP tools like kickoff does."""
    with patch.object(Agent, "get_platform_tools") as mock_platform_tools, patch.object(
        Agent, "get_mcp_tools"
    ) as mock_mcp_tools:

        class PlatformTool(BaseTool):
            name: str = "platform_tool"
            description: str = "A platform tool"

            def _run(self, **kwargs):
                return "platform result"

        class MCPTool(BaseTool):
            name: str = "mcp_tool"
            description: str = "An MCP tool"

            def _run(self, **kwargs):
                return "mcp result"

        mock_platform_tools.return_value = [PlatformTool()]
        mock_mcp_tools.return_value = [MCPTool()]

        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            apps=["test_app"],
            mcps=["test_mcp"],
            multimodal=True,
        )

        import asyncio

        asyncio.run(agent.kickoff_async("Test message"))

        mock_lite_agent.assert_called_once()
        call_kwargs = mock_lite_agent.call_args[1]
        tools = call_kwargs["tools"]

        assert any(isinstance(tool, PlatformTool) for tool in tools)
        assert any(isinstance(tool, MCPTool) for tool in tools)
        assert any(isinstance(tool, AddImageTool) for tool in tools)

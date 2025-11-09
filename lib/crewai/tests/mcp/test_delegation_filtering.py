"""Tests for delegation-aware MCP tool filtering."""

import pytest
from unittest.mock import AsyncMock, patch

from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerStdio
from crewai.mcp.filters import ToolFilterContext, create_dynamic_tool_filter
from crewai.task import Task


@pytest.fixture
def mock_tool_definitions():
    """Create mock MCP tool definitions including admin tools."""
    return [
        {
            "name": "read_file",
            "description": "Read files",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "write_file",
            "description": "Write files",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "admin_restart_service",
            "description": "Restart system service (admin only)",
            "inputSchema": {"type": "object"},
        },
        {
            "name": "admin_modify_config",
            "description": "Modify system configuration (admin only)",
            "inputSchema": {"type": "object"},
        },
    ]


@pytest.fixture
def delegation_aware_filter():
    """Create a filter that restricts delegated tasks."""

    def filter_func(context: ToolFilterContext, tool: dict) -> bool:
        tool_name = tool.get("name", "")

        # Delegated tasks cannot use admin tools
        if context.is_delegated:
            if tool_name.startswith("admin_"):
                return False

        return True

    return create_dynamic_tool_filter(filter_func)


def test_delegation_flag_set_on_delegated_task(mock_tool_definitions):
    """Test that _is_delegated flag is set when task is delegated."""
    # This would normally happen via AgentTools.delegate_work
    # We simulate it directly here

    task = Task(
        description="Delegated task",
        expected_output="Result",
    )

    # Mark as delegated (as done in base_agent_tools.py)
    task._is_delegated = True

    assert getattr(task, "_is_delegated", False) is True


def test_delegation_context_in_filter(mock_tool_definitions, delegation_aware_filter):
    """Test that delegation context is passed to filter correctly."""
    context_captured = None

    def capture_filter(context: ToolFilterContext, tool: dict) -> bool:
        nonlocal context_captured
        context_captured = context
        return True

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            tool_filter=create_dynamic_tool_filter(capture_filter),
        )

        agent = Agent(
            role="Test Agent",
            goal="Test",
            backstory="Test",
            mcps=[config],
        )

        # Load MCP tools
        agent.tools = agent.get_mcp_tools([config])

        # Non-delegated task
        normal_task = Task(
            description="Normal task",
            expected_output="Result",
            agent=agent,
        )

        agent._get_task_filtered_tools(normal_task)
        assert context_captured.is_delegated is False

        # Delegated task
        delegated_task = Task(
            description="Delegated task",
            expected_output="Result",
            agent=agent,
        )
        delegated_task._is_delegated = True

        agent._get_task_filtered_tools(delegated_task)
        assert context_captured.is_delegated is True


def test_delegated_task_restricted_tools(
    mock_tool_definitions, delegation_aware_filter
):
    """Test that delegated tasks have restricted tool access."""
    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            tool_filter=delegation_aware_filter,
        )

        agent = Agent(
            role="Senior Developer",
            goal="Complete tasks",
            backstory="Experienced developer",
            mcps=[config],
        )

        # Load MCP tools
        agent.tools = agent.get_mcp_tools([config])

        # Normal task - should have all tools
        normal_task = Task(
            description="Regular task",
            expected_output="Result",
            agent=agent,
        )

        normal_tools = agent._get_task_filtered_tools(normal_task)
        assert len(normal_tools) == 4  # All tools
        tool_names = [t.name for t in normal_tools]
        assert "python_server.py_admin_restart_service" in tool_names

        # Delegated task - should not have admin tools
        delegated_task = Task(
            description="Delegated task",
            expected_output="Result",
            agent=agent,
        )
        delegated_task._is_delegated = True

        delegated_tools = agent._get_task_filtered_tools(delegated_task)
        assert len(delegated_tools) == 2  # Only read and write
        tool_names = [t.name for t in delegated_tools]
        assert "python_server.py_read_file" in tool_names
        assert "python_server.py_write_file" in tool_names
        assert "python_server.py_admin_restart_service" not in tool_names
        assert "python_server.py_admin_modify_config" not in tool_names


def test_delegation_prevents_tool_escalation():
    """Test that delegation cannot grant tools that agent doesn't have."""
    mock_defs = [
        {"name": "safe_tool", "description": "Safe", "inputSchema": {}},
        {"name": "dangerous_tool", "description": "Dangerous", "inputSchema": {}},
    ]

    def agent_blocks_dangerous(context: ToolFilterContext, tool: dict) -> bool:
        # Agent-level: Block dangerous tool
        if not context.task:  # Agent initialization
            return tool["name"] != "dangerous_tool"

        # Task-level: Try to allow dangerous (should not work due to cascading)
        return True

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_defs)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            tool_filter=create_dynamic_tool_filter(agent_blocks_dangerous),
        )

        agent = Agent(
            role="Junior",
            goal="Test",
            backstory="Test",
            mcps=[config],
        )

        # Load MCP tools
        agent.tools = agent.get_mcp_tools([config])

        # Agent initialization blocks dangerous_tool
        agent_tools = [t.name for t in agent.tools]
        assert "python_server.py_dangerous_tool" not in agent_tools
        assert len(agent.tools) == 1

        # Task cannot get dangerous_tool even if filter says yes at task-level
        task = Task(description="Task", expected_output="Result", agent=agent)
        task_tools = agent._get_task_filtered_tools(task)

        # Can only get tools that agent already has
        assert len(task_tools) == 1
        assert task_tools[0].name == "python_server.py_safe_tool"


def test_combined_delegation_and_task_description_filtering():
    """Test complex filter combining delegation and task description."""
    mock_defs = [
        {"name": "read_file", "description": "Read", "inputSchema": {}},
        {"name": "write_file", "description": "Write", "inputSchema": {}},
        {"name": "delete_file", "description": "Delete", "inputSchema": {}},
        {"name": "admin_tool", "description": "Admin", "inputSchema": {}},
    ]

    def complex_filter(context: ToolFilterContext, tool: dict) -> bool:
        tool_name = tool["name"]

        # Admin tools never allowed for delegated tasks
        if context.is_delegated and "admin" in tool_name:
            return False

        # Task-specific restrictions
        if context.task:
            task_desc = context.task.description.lower()

            # Analysis tasks: read only
            if "analyze" in task_desc:
                return tool_name == "read_file"

            # Delegated tasks: no delete
            if context.is_delegated and "delete" in tool_name:
                return False

        return True

    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_defs)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            tool_filter=create_dynamic_tool_filter(complex_filter),
        )

        agent = Agent(
            role="Developer",
            goal="Complete tasks",
            backstory="Developer",
            mcps=[config],
        )

        # Load MCP tools
        agent.tools = agent.get_mcp_tools([config])

        # Normal analysis task
        analysis_task = Task(
            description="Analyze the codebase",
            expected_output="Report",
            agent=agent,
        )
        tools = agent._get_task_filtered_tools(analysis_task)
        assert len(tools) == 1
        assert tools[0].name == "python_server.py_read_file"

        # Delegated general task (no delete, no admin)
        delegated_task = Task(
            description="Complete the task",
            expected_output="Result",
            agent=agent,
        )
        delegated_task._is_delegated = True
        tools = agent._get_task_filtered_tools(delegated_task)
        assert len(tools) == 2  # read and write only
        tool_names = [t.name for t in tools]
        assert "python_server.py_read_file" in tool_names
        assert "python_server.py_write_file" in tool_names
        assert "python_server.py_delete_file" not in tool_names
        assert "python_server.py_admin_tool" not in tool_names

        # Delegated analysis task (read only due to both rules)
        delegated_analysis = Task(
            description="Analyze delegated work",
            expected_output="Report",
            agent=agent,
        )
        delegated_analysis._is_delegated = True
        tools = agent._get_task_filtered_tools(delegated_analysis)
        assert len(tools) == 1
        assert tools[0].name == "python_server.py_read_file"

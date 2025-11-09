"""Tests for task-level MCP tool filtering."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from crewai.agent.core import Agent
from crewai.mcp.config import MCPServerStdio
from crewai.mcp.filters import ToolFilterContext, create_dynamic_tool_filter
from crewai.task import Task
from crewai.tools.base_tool import BaseTool


@pytest.fixture
def mock_tool_definitions():
    """Create mock MCP tool definitions with different capabilities."""
    return [
        {
            "name": "read_file",
            "description": "Read files from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write files to the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "delete_file",
            "description": "Delete files from the filesystem",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for files",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    ]


@pytest.fixture
def task_aware_filter():
    """Create a task-aware filter function."""

    def filter_func(context: ToolFilterContext, tool: dict) -> bool:
        tool_name = tool.get("name", "")

        # Agent-level: Block delete for junior developers
        if context.agent.role == "Junior Developer":
            if "delete" in tool_name:
                return False

        # Task-level: Restrict based on task description
        if context.task:
            task_desc = context.task.description.lower()

            if "analyze" in task_desc or "review" in task_desc:
                # Analysis tasks only get read and search tools
                # Check if tool name ends with the base name (handles server prefix)
                return tool_name.endswith("read_file") or tool_name.endswith("search_files")

            if "implement" in task_desc:
                # Implementation tasks: no delete
                return "delete" not in tool_name

        return True

    return create_dynamic_tool_filter(filter_func)


def test_agent_level_filtering_only(mock_tool_definitions):
    """Test that agent-level filtering works without task context."""
    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        def agent_filter(context: ToolFilterContext, tool: dict) -> bool:
            # Block delete tools for all agents
            return "delete" not in tool.get("name", "")

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            tool_filter=create_dynamic_tool_filter(agent_filter),
        )

        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            mcps=[config],
        )

        tools = agent.get_mcp_tools([config])

        # Should get all tools except delete_file
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "python_server.py_read_file" in tool_names
        assert "python_server.py_write_file" in tool_names
        assert "python_server.py_search_files" in tool_names
        assert "python_server.py_delete_file" not in tool_names


def test_task_level_filtering_cascades(mock_tool_definitions, task_aware_filter):
    """Test that task-level filtering further restricts agent-level tools."""
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
            tool_filter=task_aware_filter,
        )

        agent = Agent(
            role="Junior Developer",
            goal="Complete coding tasks",
            backstory="Learning developer",
            mcps=[config],
        )

        # Agent-level tools (delete blocked)
        agent_tools = agent.get_mcp_tools([config])
        assert len(agent_tools) == 3  # All except delete

        # Assign tools to agent for task-level filtering
        agent.tools = agent_tools

        # Create analysis task
        analysis_task = Task(
            description="Analyze the authentication code for security issues",
            expected_output="Security analysis report",
            agent=agent,
        )

        # Get task-filtered tools
        task_tools = agent._get_task_filtered_tools(analysis_task)

        # Should only have read and search tools
        assert len(task_tools) == 2
        tool_names = [t.name for t in task_tools]
        assert "python_server.py_read_file" in tool_names
        assert "python_server.py_search_files" in tool_names
        assert "python_server.py_write_file" not in tool_names


def test_task_level_filtering_different_tasks(
    mock_tool_definitions, task_aware_filter
):
    """Test that different tasks get different tool subsets."""
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
            tool_filter=task_aware_filter,
        )

        agent = Agent(
            role="Junior Developer",
            goal="Complete tasks",
            backstory="Developer",
            mcps=[config],
        )

        # Load and assign MCP tools
        agent.tools = agent.get_mcp_tools([config])

        # Analysis task
        analysis_task = Task(
            description="Analyze code structure",
            expected_output="Analysis report",
            agent=agent,
        )
        analysis_tools = agent._get_task_filtered_tools(analysis_task)
        assert len(analysis_tools) == 2  # read, search

        # Implementation task
        impl_task = Task(
            description="Implement new login feature",
            expected_output="Working feature",
            agent=agent,
        )
        impl_tools = agent._get_task_filtered_tools(impl_task)
        assert len(impl_tools) == 3  # read, write, search (no delete)

        # Generic task (no keywords)
        generic_task = Task(
            description="Complete the task",
            expected_output="Result",
            agent=agent,
        )
        generic_tools = agent._get_task_filtered_tools(generic_task)
        assert len(generic_tools) == 3  # All except delete (agent-level block)


def test_task_filtering_preserves_non_mcp_tools(mock_tool_definitions):
    """Test that non-MCP tools pass through task filtering unchanged."""
    from crewai.tools import tool

    @tool
    def custom_tool(query: str) -> str:
        """A custom non-MCP tool."""
        return f"Result for {query}"

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
            tool_filter=lambda ctx, tool: tool["name"] == "read_file",
        )

        agent = Agent(
            role="Test Agent",
            goal="Test",
            backstory="Test",
            mcps=[config],
            tools=[custom_tool],
        )

        # Load MCP tools and combine with existing tools
        mcp_tools = agent.get_mcp_tools([config])
        agent.tools = [custom_tool] + mcp_tools

        task = Task(
            description="Test task",
            expected_output="Result",
            agent=agent,
        )

        filtered_tools = agent._get_task_filtered_tools(task)

        # Should have custom tool + 1 MCP tool (read_file)
        assert len(filtered_tools) == 2
        tool_names = [t.name for t in filtered_tools]
        assert "custom_tool" in tool_names
        assert "python_server.py_read_file" in tool_names


def test_no_filter_allows_all_tools(mock_tool_definitions):
    """Test that absence of filter allows all tools."""
    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        config = MCPServerStdio(command="python", args=["server.py"])

        agent = Agent(
            role="Test Agent",
            goal="Test",
            backstory="Test",
            mcps=[config],
        )

        # Load and assign MCP tools
        agent.tools = agent.get_mcp_tools([config])

        task = Task(description="Test task", expected_output="Result", agent=agent)

        tools = agent._get_task_filtered_tools(task)

        # Should have all 4 tools
        assert len(tools) == 4


def test_static_filter_fallback(mock_tool_definitions):
    """Test that static filters work as fallback."""
    with patch("crewai.agent.core.MCPClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=mock_tool_definitions)
        mock_client.connected = False
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client_class.return_value = mock_client

        # Static filter (1 param instead of 2)
        def static_filter(tool: dict) -> bool:
            return "delete" not in tool.get("name", "")

        config = MCPServerStdio(
            command="python",
            args=["server.py"],
            tool_filter=static_filter,
        )

        agent = Agent(
            role="Test Agent",
            goal="Test",
            backstory="Test",
            mcps=[config],
        )

        # Load and assign MCP tools
        agent.tools = agent.get_mcp_tools([config])

        task = Task(description="Test task", expected_output="Result", agent=agent)

        tools = agent._get_task_filtered_tools(task)

        # Should have 3 tools (all except delete)
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "python_server.py_delete_file" not in tool_names


def test_tool_filter_context_has_task(mock_tool_definitions):
    """Test that ToolFilterContext has task field populated."""
    context_captured = None

    def capture_context_filter(context: ToolFilterContext, tool: dict) -> bool:
        nonlocal context_captured
        # Only capture context when task is present (task-level filtering)
        if context.task is not None:
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
            tool_filter=create_dynamic_tool_filter(capture_context_filter),
        )

        agent = Agent(
            role="Test Agent",
            goal="Test",
            backstory="Test",
            mcps=[config],
        )

        # Load MCP tools into agent
        agent.tools = agent.get_mcp_tools([config])

        task = Task(
            description="Test task with specific description",
            expected_output="Result",
            agent=agent,
        )

        agent._get_task_filtered_tools(task)

        # Verify context was populated
        assert context_captured is not None
        assert context_captured.task is task
        assert context_captured.task_description == "Test task with specific description"
        assert context_captured.has_task_context is True
        assert context_captured.is_delegated is False


def test_convenience_properties(mock_tool_definitions):
    """Test ToolFilterContext convenience properties."""
    from crewai.mcp.filters import ToolFilterContext

    task = Task(
        description="Test description",
        expected_output="Expected output text",
    )

    # Rebuild model with Task in scope to resolve forward references
    ToolFilterContext.model_rebuild(_types_namespace={"Task": Task})

    context = ToolFilterContext(
        agent=MagicMock(role="Test"),
        server_name="test_server",
        task=task,
        is_delegated=True,
    )

    assert context.task_description == "Test description"
    assert context.task_expected_output == "Expected output text"
    assert context.has_task_context is True

    # Test without task
    context_no_task = ToolFilterContext(
        agent=MagicMock(role="Test"),
        server_name="test_server",
    )

    assert context_no_task.task_description == ""
    assert context_no_task.task_expected_output == ""
    assert context_no_task.has_task_context is False

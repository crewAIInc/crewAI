"""Tests for MCPToolWrapper progress and headers support."""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import MCPToolProgressEvent
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


@pytest.fixture(autouse=True)
def stub_mcp_modules(monkeypatch):
    """Stub the mcp modules in sys.modules to avoid import errors in CI."""
    mcp = types.ModuleType("mcp")
    mcp_client = types.ModuleType("mcp.client")
    mcp_streamable_http = types.ModuleType("mcp.client.streamable_http")
    
    mcp.ClientSession = MagicMock()
    
    async def fake_streamablehttp_client(*args, **kwargs):
        """Mock streamablehttp_client context manager."""
        class MockContextManager:
            async def __aenter__(self):
                return (AsyncMock(), AsyncMock(), AsyncMock())
            
            async def __aexit__(self, *exc):
                pass
        
        return MockContextManager()
    
    mcp_streamable_http.streamablehttp_client = fake_streamablehttp_client
    
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", mcp_client)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", mcp_streamable_http)


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP ClientSession."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock()
    return session


@pytest.fixture
def mock_streamable_client(mock_mcp_session):
    """Create a mock streamablehttp_client context manager."""
    async def mock_client(*args, **kwargs):
        read = AsyncMock()
        write = AsyncMock()
        close = AsyncMock()
        
        class MockContextManager:
            async def __aenter__(self):
                return (read, write, close)
            
            async def __aexit__(self, *args):
                pass
        
        return MockContextManager()
    
    return mock_client


@pytest.fixture
def mock_agent():
    """Create a mock agent with id and role."""
    agent = Mock()
    agent.id = "test-agent-id"
    agent.role = "Test Agent"
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task with id and description."""
    task = Mock()
    task.id = "test-task-id"
    task.description = "Test Task Description"
    task.name = None
    return task


class TestMCPToolWrapperProgress:
    """Test suite for MCP tool wrapper progress notifications."""

    def test_wrapper_initialization_with_progress_callback(self):
        """Test that MCPToolWrapper can be initialized with progress callback."""
        callback = Mock()
        
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
            progress_callback=callback,
        )
        
        assert wrapper._progress_callback == callback
        assert wrapper.name == "test_server_test_tool"

    def test_wrapper_initialization_without_progress_callback(self):
        """Test that MCPToolWrapper works without progress callback."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
        )
        
        assert wrapper._progress_callback is None

    def test_wrapper_initialization_with_agent_and_task(self, mock_agent, mock_task):
        """Test that MCPToolWrapper can be initialized with agent and task context."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
            agent=mock_agent,
            task=mock_task,
        )
        
        assert wrapper._agent == mock_agent
        assert wrapper._task == mock_task

    @pytest.mark.asyncio
    async def test_progress_handler_called_during_execution(self, mock_agent, mock_task):
        """Test that progress callback is invoked when MCP server sends progress."""
        progress_callback = Mock()
        
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
            progress_callback=progress_callback,
            agent=mock_agent,
            task=mock_task,
        )
        
        mock_result = Mock()
        mock_result.content = [Mock(text="Test result")]
        
        with patch("crewai.tools.mcp_tool_wrapper.streamablehttp_client") as mock_client, \
             patch("crewai.tools.mcp_tool_wrapper.ClientSession") as mock_session_class:
            
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            mock_session.on_progress = None
            
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock()
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=(Mock(), Mock(), Mock()))
            mock_client.return_value.__aexit__ = AsyncMock()
            
            result = await wrapper._execute_tool(test_arg="test_value")
            
            assert result == "Test result"
            
            assert mock_session.on_progress is not None

    @pytest.mark.asyncio
    async def test_progress_event_emission(self, mock_agent, mock_task):
        """Test that MCPToolProgressEvent is emitted when progress is reported."""
        events_received = []
        
        def event_handler(source, event):
            if isinstance(event, MCPToolProgressEvent):
                events_received.append(event)
        
        crewai_event_bus.register_handler(MCPToolProgressEvent, event_handler)
        
        try:
            wrapper = MCPToolWrapper(
                mcp_server_params={"url": "https://example.com/mcp"},
                tool_name="test_tool",
                tool_schema={"description": "Test tool"},
                server_name="test_server",
                progress_callback=Mock(),
                agent=mock_agent,
                task=mock_task,
            )
            
            wrapper._emit_progress_event(50.0, 100.0, "Processing...")
            
            await asyncio.sleep(0.1)
            
            assert len(events_received) == 1
            event = events_received[0]
            assert event.tool_name == "test_tool"
            assert event.server_name == "test_server"
            assert event.progress == 50.0
            assert event.total == 100.0
            assert event.message == "Processing..."
            assert event.agent_id == "test-agent-id"
            assert event.agent_role == "Test Agent"
            assert event.task_id == "test-task-id"
            assert event.task_name == "Test Task Description"
        
        finally:
            crewai_event_bus._sync_handlers.pop(MCPToolProgressEvent, None)

    def test_progress_event_without_agent_context(self):
        """Test that progress events work without agent context."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
            progress_callback=Mock(),
        )
        
        wrapper._emit_progress_event(25.0, None, "Starting...")

    def test_progress_event_without_task_context(self, mock_agent):
        """Test that progress events work without task context."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
            progress_callback=Mock(),
            agent=mock_agent,
        )
        
        wrapper._emit_progress_event(75.0, 100.0, None)


class TestMCPToolWrapperHeaders:
    """Test suite for MCP tool wrapper headers support."""

    def test_wrapper_initialization_with_headers(self):
        """Test that MCPToolWrapper accepts headers in server params."""
        headers = {"Authorization": "Bearer token123", "X-Client-ID": "test-client"}
        
        wrapper = MCPToolWrapper(
            mcp_server_params={
                "url": "https://example.com/mcp",
                "headers": headers,
            },
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
        )
        
        assert wrapper.mcp_server_params["headers"] == headers

    @pytest.mark.asyncio
    async def test_headers_passed_to_transport(self):
        """Test that headers are passed to streamablehttp_client."""
        headers = {"Authorization": "Bearer token123"}
        
        wrapper = MCPToolWrapper(
            mcp_server_params={
                "url": "https://example.com/mcp",
                "headers": headers,
            },
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
        )
        
        mock_result = Mock()
        mock_result.content = [Mock(text="Test result")]
        
        with patch("crewai.tools.mcp_tool_wrapper.streamablehttp_client") as mock_client, \
             patch("crewai.tools.mcp_tool_wrapper.ClientSession") as mock_session_class:
            
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock()
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=(Mock(), Mock(), Mock()))
            mock_client.return_value.__aexit__ = AsyncMock()
            
            await wrapper._execute_tool(test_arg="test_value")
            
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert call_args[0][0] == "https://example.com/mcp"
            assert "headers" in call_args[1]
            assert call_args[1]["headers"] == headers

    @pytest.mark.asyncio
    async def test_no_headers_when_not_configured(self):
        """Test that headers are not passed when not configured."""
        wrapper = MCPToolWrapper(
            mcp_server_params={"url": "https://example.com/mcp"},
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
        )
        
        mock_result = Mock()
        mock_result.content = [Mock(text="Test result")]
        
        with patch("crewai.tools.mcp_tool_wrapper.streamablehttp_client") as mock_client, \
             patch("crewai.tools.mcp_tool_wrapper.ClientSession") as mock_session_class:
            
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock()
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=(Mock(), Mock(), Mock()))
            mock_client.return_value.__aexit__ = AsyncMock()
            
            await wrapper._execute_tool(test_arg="test_value")
            
            mock_client.assert_called_once()
            call_args = mock_client.call_args
            assert "headers" not in call_args[1] or call_args[1].get("headers") is None


class TestMCPToolWrapperIntegration:
    """Integration tests for MCP tool wrapper with progress and headers."""

    @pytest.mark.asyncio
    async def test_full_execution_with_progress_and_headers(self, mock_agent, mock_task):
        """Test complete execution flow with both progress and headers."""
        progress_calls = []
        
        def progress_callback(progress, total, message):
            progress_calls.append((progress, total, message))
        
        headers = {"Authorization": "Bearer test-token"}
        
        wrapper = MCPToolWrapper(
            mcp_server_params={
                "url": "https://example.com/mcp",
                "headers": headers,
            },
            tool_name="test_tool",
            tool_schema={"description": "Test tool"},
            server_name="test_server",
            progress_callback=progress_callback,
            agent=mock_agent,
            task=mock_task,
        )
        
        mock_result = Mock()
        mock_result.content = [Mock(text="Test result")]
        
        with patch("crewai.tools.mcp_tool_wrapper.streamablehttp_client") as mock_client, \
             patch("crewai.tools.mcp_tool_wrapper.ClientSession") as mock_session_class:
            
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            mock_session.on_progress = None
            
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock()
            
            mock_client.return_value.__aenter__ = AsyncMock(return_value=(Mock(), Mock(), Mock()))
            mock_client.return_value.__aexit__ = AsyncMock()
            
            result = await wrapper._execute_tool(test_arg="test_value")
            
            assert result == "Test result"
            
            call_args = mock_client.call_args
            assert call_args[1]["headers"] == headers
            
            assert mock_session.on_progress is not None

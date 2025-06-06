"""Tests for A2A server utilities."""

import pytest
from unittest.mock import Mock, patch

try:
    from crewai.a2a import start_a2a_server, create_a2a_app
    from a2a.server.agent_execution.agent_executor import AgentExecutor
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A integration not available")
class TestA2AServer:
    """Test cases for A2A server utilities."""
    
    @pytest.fixture
    def mock_agent_executor(self):
        """Create a mock AgentExecutor."""
        return Mock(spec=AgentExecutor)
    
    @patch('uvicorn.run')
    @patch('crewai.a2a.server.create_a2a_app')
    def test_start_a2a_server_default(self, mock_create_app, mock_uvicorn_run, mock_agent_executor):
        """Test starting A2A server with default parameters."""
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        
        start_a2a_server(mock_agent_executor)
        
        mock_create_app.assert_called_once_with(
            mock_agent_executor, 
            transport="starlette"
        )
        
        mock_uvicorn_run.assert_called_once_with(
            mock_app, 
            host="localhost", 
            port=10001
        )
    
    @patch('uvicorn.run')
    @patch('crewai.a2a.server.create_a2a_app')
    def test_start_a2a_server_custom(self, mock_create_app, mock_uvicorn_run, mock_agent_executor):
        """Test starting A2A server with custom parameters."""
        mock_app = Mock()
        mock_create_app.return_value = mock_app
        
        start_a2a_server(
            mock_agent_executor,
            host="0.0.0.0",
            port=8080,
            transport="fastapi"
        )
        
        mock_create_app.assert_called_once_with(
            mock_agent_executor,
            transport="fastapi"
        )
        
        mock_uvicorn_run.assert_called_once_with(
            mock_app,
            host="0.0.0.0",
            port=8080
        )
    
    @patch('crewai.a2a.server.A2AStarletteApplication')
    @patch('crewai.a2a.server.DefaultRequestHandler')
    @patch('crewai.a2a.server.InMemoryTaskStore')
    def test_create_a2a_app_starlette(self, mock_task_store_class, mock_handler_class, mock_app_class, mock_agent_executor):
        """Test creating A2A app with Starlette transport."""
        mock_handler = Mock()
        mock_app_instance = Mock()
        mock_built_app = Mock()
        mock_task_store = Mock()
        
        mock_task_store_class.return_value = mock_task_store
        mock_handler_class.return_value = mock_handler
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.build.return_value = mock_built_app
        
        result = create_a2a_app(mock_agent_executor, transport="starlette")
        
        mock_task_store_class.assert_called_once()
        mock_handler_class.assert_called_once_with(mock_agent_executor, mock_task_store)
        mock_app_class.assert_called_once()
        mock_app_instance.build.assert_called_once()
        
        assert result == mock_built_app
    
    def test_create_a2a_app_fastapi(self, mock_agent_executor):
        """Test creating A2A app with FastAPI transport raises error."""
        with pytest.raises(ValueError, match="FastAPI transport is not available"):
            create_a2a_app(
                mock_agent_executor,
                transport="fastapi",
                agent_name="Custom Agent",
                agent_description="Custom description"
            )
    
    @patch('crewai.a2a.server.A2AStarletteApplication')
    @patch('crewai.a2a.server.DefaultRequestHandler')
    @patch('crewai.a2a.server.InMemoryTaskStore')
    def test_create_a2a_app_default_transport(self, mock_task_store_class, mock_handler_class, mock_app_class, mock_agent_executor):
        """Test creating A2A app with default transport."""
        mock_handler = Mock()
        mock_app_instance = Mock()
        mock_built_app = Mock()
        mock_task_store = Mock()
        
        mock_task_store_class.return_value = mock_task_store
        mock_handler_class.return_value = mock_handler
        mock_app_class.return_value = mock_app_instance
        mock_app_instance.build.return_value = mock_built_app
        
        result = create_a2a_app(mock_agent_executor)
        
        mock_task_store_class.assert_called_once()
        mock_handler_class.assert_called_once_with(mock_agent_executor, mock_task_store)
        mock_app_class.assert_called_once()
        assert result == mock_built_app


@pytest.mark.skipif(A2A_AVAILABLE, reason="Testing import error handling")
def test_server_import_error_handling():
    """Test that import errors are handled gracefully when A2A is not available."""
    with pytest.raises(ImportError, match="A2A integration requires"):
        from crewai.a2a.server import start_a2a_server

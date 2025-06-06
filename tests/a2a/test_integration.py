"""Integration tests for CrewAI A2A functionality."""

import pytest
from unittest.mock import Mock, patch

from crewai import Agent, Crew, Task

try:
    from crewai.a2a import CrewAgentExecutor, create_a2a_app
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A integration not available")
class TestA2AIntegration:
    """Integration tests for A2A functionality."""
    
    @pytest.fixture
    def sample_crew(self):
        """Create a sample crew for integration testing."""
        from unittest.mock import Mock
        mock_crew = Mock()
        mock_crew.agents = []
        mock_crew.tasks = []
        return mock_crew
    
    def test_end_to_end_integration(self, sample_crew):
        """Test end-to-end A2A integration."""
        executor = CrewAgentExecutor(sample_crew)
        
        assert executor.crew == sample_crew
        assert isinstance(executor.supported_content_types, list)
        
        with patch('crewai.a2a.server.A2AStarletteApplication') as mock_app_class:
            with patch('crewai.a2a.server.DefaultRequestHandler') as mock_handler_class:
                with patch('crewai.a2a.server.InMemoryTaskStore') as mock_task_store_class:
                    mock_handler = Mock()
                    mock_app_instance = Mock()
                    mock_built_app = Mock()
                    mock_task_store = Mock()
                    
                    mock_task_store_class.return_value = mock_task_store
                    mock_handler_class.return_value = mock_handler
                    mock_app_class.return_value = mock_app_instance
                    mock_app_instance.build.return_value = mock_built_app
                    
                    app = create_a2a_app(executor)
                    
                    mock_task_store_class.assert_called_once()
                    mock_handler_class.assert_called_once_with(executor, mock_task_store)
                    mock_app_class.assert_called_once()
                    assert app == mock_built_app
    
    def test_crew_with_multiple_agents(self):
        """Test A2A integration with multi-agent crew."""
        from unittest.mock import Mock
        crew = Mock()
        crew.agents = [Mock(), Mock()]
        crew.tasks = [Mock(), Mock()]
        
        executor = CrewAgentExecutor(crew)
        assert executor.crew == crew
        assert len(executor.crew.agents) == 2
        assert len(executor.crew.tasks) == 2
    
    def test_custom_content_types(self, sample_crew):
        """Test A2A integration with custom content types."""
        custom_types = ['text', 'application/json', 'image/png']
        executor = CrewAgentExecutor(
            sample_crew,
            supported_content_types=custom_types
        )
        
        assert executor.supported_content_types == custom_types
    
    @patch('uvicorn.run')
    def test_server_startup_integration(self, mock_uvicorn_run, sample_crew):
        """Test server startup integration."""
        from crewai.a2a import start_a2a_server
        
        executor = CrewAgentExecutor(sample_crew)
        
        with patch('crewai.a2a.server.create_a2a_app') as mock_create_app:
            mock_app = Mock()
            mock_create_app.return_value = mock_app
            
            start_a2a_server(
                executor,
                host="127.0.0.1",
                port=9999,
                transport="starlette"
            )
            
            mock_create_app.assert_called_once_with(
                executor,
                transport="starlette"
            )
            mock_uvicorn_run.assert_called_once_with(
                mock_app,
                host="127.0.0.1",
                port=9999
            )


def test_optional_import_in_main_module():
    """Test that A2A classes are optionally imported in main module."""
    import crewai
    
    if A2A_AVAILABLE:
        assert hasattr(crewai, 'CrewAgentExecutor')
        assert hasattr(crewai, 'start_a2a_server')
        assert hasattr(crewai, 'create_a2a_app')
        
        assert 'CrewAgentExecutor' in crewai.__all__
        assert 'start_a2a_server' in crewai.__all__
        assert 'create_a2a_app' in crewai.__all__
    else:
        assert not hasattr(crewai, 'CrewAgentExecutor')
        assert not hasattr(crewai, 'start_a2a_server')
        assert not hasattr(crewai, 'create_a2a_app')

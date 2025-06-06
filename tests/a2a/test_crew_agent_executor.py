"""Tests for CrewAgentExecutor class."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from crewai.crews.crew_output import CrewOutput

try:
    from crewai.a2a import CrewAgentExecutor
    from a2a.server.agent_execution import RequestContext
    from a2a.server.events import EventQueue
    pass  # Imports handled in test methods as needed
    from a2a.utils.errors import ServerError
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A integration not available")
class TestCrewAgentExecutor:
    """Test cases for CrewAgentExecutor."""
    
    @pytest.fixture
    def sample_crew(self):
        """Create a sample crew for testing."""
        from unittest.mock import Mock
        mock_crew = Mock()
        mock_crew.agents = []
        mock_crew.tasks = []
        return mock_crew
    
    @pytest.fixture
    def crew_executor(self, sample_crew):
        """Create a CrewAgentExecutor for testing."""
        return CrewAgentExecutor(sample_crew)
    
    @pytest.fixture
    def mock_context(self):
        """Create a mock RequestContext."""
        from a2a.types import Message, Part, TextPart
        context = Mock(spec=RequestContext)
        context.task_id = "test-task-123"
        context.context_id = "test-context-456"
        context.message = Message(
            messageId="msg-123",
            taskId="test-task-123",
            contextId="test-context-456",
            role="user",
            parts=[Part(root=TextPart(text="Test message"))]
        )
        context.get_user_input.return_value = "Test query"
        return context
    
    @pytest.fixture
    def mock_event_queue(self):
        """Create a mock EventQueue."""
        return Mock(spec=EventQueue)
    
    def test_init(self, sample_crew):
        """Test CrewAgentExecutor initialization."""
        executor = CrewAgentExecutor(sample_crew)
        
        assert executor.crew == sample_crew
        assert executor.supported_content_types == ['text', 'text/plain']
        assert executor._running_tasks == {}
    
    def test_init_with_custom_content_types(self, sample_crew):
        """Test CrewAgentExecutor initialization with custom content types."""
        custom_types = ['text', 'application/json']
        executor = CrewAgentExecutor(sample_crew, supported_content_types=custom_types)
        
        assert executor.supported_content_types == custom_types
    
    @pytest.mark.asyncio
    async def test_execute_success(self, crew_executor, mock_context, mock_event_queue):
        """Test successful crew execution."""
        mock_output = CrewOutput(raw="Test response", json_dict=None)
        
        with patch.object(crew_executor, '_execute_crew_async', return_value=mock_output):
            await crew_executor.execute(mock_context, mock_event_queue)
        
        mock_event_queue.enqueue_event.assert_called_once()
        
        assert len(crew_executor._running_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_validation_error(self, crew_executor, mock_event_queue):
        """Test execution with validation error."""
        bad_context = Mock(spec=RequestContext)
        bad_context.get_user_input.return_value = ""
        
        with pytest.raises(ServerError):
            await crew_executor.execute(bad_context, mock_event_queue)
    
    @pytest.mark.asyncio
    async def test_execute_with_crew_error(self, crew_executor, mock_context, mock_event_queue):
        """Test execution when crew raises an error."""
        with patch.object(crew_executor, '_execute_crew_async', side_effect=Exception("Crew error")):
            with pytest.raises(ServerError):
                await crew_executor.execute(mock_context, mock_event_queue)
        
        mock_event_queue.enqueue_event.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_existing_task(self, crew_executor, mock_event_queue):
        """Test cancelling an existing task."""
        cancel_context = Mock(spec=RequestContext)
        cancel_context.task_id = "test-task-123"
        
        async def dummy_task():
            await asyncio.sleep(10)
        
        mock_task = asyncio.create_task(dummy_task())
        from crewai.a2a.crew_agent_executor import TaskInfo
        from datetime import datetime
        task_info = TaskInfo(task=mock_task, started_at=datetime.now())
        crew_executor._running_tasks["test-task-123"] = task_info
        
        result = await crew_executor.cancel(cancel_context, mock_event_queue)
        
        assert result is None
        assert "test-task-123" not in crew_executor._running_tasks
        assert mock_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, crew_executor, mock_event_queue):
        """Test cancelling a task that doesn't exist."""
        cancel_context = Mock(spec=RequestContext)
        cancel_context.task_id = "nonexistent-task"
        
        with pytest.raises(ServerError):
            await crew_executor.cancel(cancel_context, mock_event_queue)
    
    def test_convert_output_to_parts_with_raw(self, crew_executor):
        """Test converting crew output with raw content to A2A parts."""
        output = Mock()
        output.raw = "Test response"
        output.json_dict = None
        parts = crew_executor._convert_output_to_parts(output)
        
        assert len(parts) == 1
        assert parts[0].root.text == "Test response"
    
    def test_convert_output_to_parts_with_json(self, crew_executor):
        """Test converting crew output with JSON data to A2A parts."""
        output = Mock()
        output.raw = "Test response"
        output.json_dict = {"key": "value"}
        parts = crew_executor._convert_output_to_parts(output)
        
        assert len(parts) == 2
        assert parts[0].root.text == "Test response"
        assert '"key": "value"' in parts[1].root.text
    
    def test_convert_output_to_parts_empty(self, crew_executor):
        """Test converting empty crew output to A2A parts."""
        output = ""
        parts = crew_executor._convert_output_to_parts(output)
        
        assert len(parts) == 1
        assert parts[0].root.text == "Crew execution completed successfully"
    
    def test_validate_request_valid(self, crew_executor, mock_context):
        """Test request validation with valid input."""
        error = crew_executor._validate_request(mock_context)
        assert error is None
    
    def test_validate_request_empty_input(self, crew_executor):
        """Test request validation with empty input."""
        context = Mock(spec=RequestContext)
        context.get_user_input.return_value = ""
        
        error = crew_executor._validate_request(context)
        assert error == "Empty or missing user input"
    
    def test_validate_request_whitespace_input(self, crew_executor):
        """Test request validation with whitespace-only input."""
        context = Mock(spec=RequestContext)
        context.get_user_input.return_value = "   \n\t  "
        
        error = crew_executor._validate_request(context)
        assert error == "Empty or missing user input"
    
    def test_validate_request_exception(self, crew_executor):
        """Test request validation when get_user_input raises exception."""
        context = Mock(spec=RequestContext)
        context.get_user_input.side_effect = Exception("Input error")
        
        error = crew_executor._validate_request(context)
        assert "Failed to extract user input" in error


@pytest.mark.skipif(A2A_AVAILABLE, reason="Testing import error handling")
def test_import_error_handling():
    """Test that import errors are handled gracefully when A2A is not available."""
    with pytest.raises(ImportError, match="A2A integration requires"):
        pass

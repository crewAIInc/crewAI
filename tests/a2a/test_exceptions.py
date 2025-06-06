"""Tests for A2A custom exceptions."""

import pytest

try:
    from crewai.a2a.crew_agent_executor import (
        A2AServerError,
        TransportError,
        ExecutionError
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A integration not available")
class TestA2AExceptions:
    """Test A2A custom exception classes."""
    
    def test_a2a_server_error_base(self):
        """Test A2AServerError base exception."""
        error = A2AServerError("Base error message")
        
        assert str(error) == "Base error message"
        assert isinstance(error, Exception)
    
    def test_transport_error_inheritance(self):
        """Test TransportError inherits from A2AServerError."""
        error = TransportError("Transport configuration failed")
        
        assert str(error) == "Transport configuration failed"
        assert isinstance(error, A2AServerError)
        assert isinstance(error, Exception)
    
    def test_execution_error_inheritance(self):
        """Test ExecutionError inherits from A2AServerError."""
        error = ExecutionError("Crew execution failed")
        
        assert str(error) == "Crew execution failed"
        assert isinstance(error, A2AServerError)
        assert isinstance(error, Exception)
    
    def test_exception_raising(self):
        """Test that exceptions can be raised and caught."""
        with pytest.raises(TransportError) as exc_info:
            raise TransportError("Test transport error")
        
        assert str(exc_info.value) == "Test transport error"
        
        with pytest.raises(ExecutionError) as exc_info:
            raise ExecutionError("Test execution error")
        
        assert str(exc_info.value) == "Test execution error"
        
        with pytest.raises(A2AServerError):
            raise TransportError("Should be caught as base class")

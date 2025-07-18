"""
Tests for custom endpoint error handling (issue #3165).

These tests verify that CrewAI properly propagates specific error details
from custom OpenAI-compatible endpoints instead of showing generic "LLM Failed" errors.
"""

import pytest
from unittest.mock import patch, MagicMock
from crewai.llm import LLM
from crewai.utilities.events.llm_events import LLMCallFailedEvent
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
import requests


class TestCustomEndpointErrorHandling:
    """Test error handling for custom OpenAI-compatible endpoints."""

    def test_connection_error_preserves_details(self):
        """Test that connection errors preserve specific error details."""
        custom_llm = LLM(
            model="gpt-3.5-turbo",
            base_url="https://non-existent-endpoint.example.com/v1",
            api_key="fake-api-key"
        )
        
        with patch('litellm.completion') as mock_completion:
            mock_completion.side_effect = requests.exceptions.ConnectionError(
                "Failed to establish a new connection: [Errno -2] Name or service not known"
            )
            
            with pytest.raises(requests.exceptions.ConnectionError) as exc_info:
                custom_llm.call("Hello world")
            
            assert "Name or service not known" in str(exc_info.value)

    def test_authentication_error_preserves_details(self):
        """Test that authentication errors preserve specific error details."""
        custom_llm = LLM(
            model="gpt-3.5-turbo",
            base_url="https://api.openai.com/v1",
            api_key="invalid-api-key"
        )
        
        with patch('litellm.completion') as mock_completion:
            mock_completion.side_effect = Exception(
                "AuthenticationError: Incorrect API key provided"
            )
            
            with pytest.raises(Exception) as exc_info:
                custom_llm.call("Hello world")
            
            assert "AuthenticationError" in str(exc_info.value)
            assert "Incorrect API key" in str(exc_info.value)

    def test_llm_call_failed_event_enhanced_fields(self):
        """Test that LLMCallFailedEvent includes enhanced error information."""
        custom_llm = LLM(
            model="gpt-3.5-turbo",
            base_url="https://custom-endpoint.example.com/v1",
            api_key="test-key"
        )
        
        captured_events = []
        
        def capture_event(sender, event):
            captured_events.append(event)
        
        with patch('crewai.utilities.events.crewai_event_bus.crewai_event_bus.emit', side_effect=capture_event):
            with patch('litellm.completion') as mock_completion:
                mock_completion.side_effect = requests.exceptions.ConnectionError(
                    "Connection failed"
                )
                
                with pytest.raises(requests.exceptions.ConnectionError):
                    custom_llm.call("Hello world")
        
        assert len(captured_events) == 2  # Started and Failed events
        failed_event = captured_events[1]
        assert isinstance(failed_event, LLMCallFailedEvent)
        assert failed_event.error_type == "ConnectionError"
        assert failed_event.original_error == "Connection failed"
        assert failed_event.endpoint_info is not None
        assert failed_event.endpoint_info["base_url"] == "https://custom-endpoint.example.com/v1"
        assert failed_event.endpoint_info["model"] == "gpt-3.5-turbo"

    def test_console_formatter_displays_enhanced_error_info(self):
        """Test that console formatter displays enhanced error information."""
        formatter = ConsoleFormatter(verbose=True)
        
        mock_event = MagicMock()
        mock_event.error_type = "ConnectionError"
        mock_event.endpoint_info = {
            "base_url": "https://custom-endpoint.example.com/v1",
            "model": "gpt-3.5-turbo"
        }
        
        captured_output = []
        
        def mock_print_panel(content, title, style):
            captured_output.append(str(content))
        
        formatter.print_panel = mock_print_panel
        
        formatter.handle_llm_call_failed(
            tool_branch=None,
            error="Connection failed",
            crew_tree=None,
            event=mock_event
        )
        
        output = captured_output[0]
        assert "Error Type: ConnectionError" in output
        assert "Endpoint: https://custom-endpoint.example.com/v1" in output
        assert "Model: gpt-3.5-turbo" in output
        assert "Connection failed" in output

    def test_backward_compatibility_without_enhanced_fields(self):
        """Test that console formatter works without enhanced fields for backward compatibility."""
        formatter = ConsoleFormatter(verbose=True)
        
        captured_output = []
        
        def mock_print_panel(content, title, style):
            captured_output.append(str(content))
        
        formatter.print_panel = mock_print_panel
        
        formatter.handle_llm_call_failed(
            tool_branch=None,
            error="Generic error message",
            crew_tree=None,
            event=None
        )
        
        output = captured_output[0]
        assert "❌ LLM Call Failed" in output
        assert "Generic error message" in output
        assert "Error Type:" not in output
        assert "Endpoint:" not in output

    def test_streaming_response_error_handling(self):
        """Test that streaming responses also preserve error details."""
        custom_llm = LLM(
            model="gpt-3.5-turbo",
            base_url="https://custom-endpoint.example.com/v1",
            api_key="test-key",
            stream=True
        )
        
        with patch('litellm.completion') as mock_completion:
            mock_completion.side_effect = requests.exceptions.ConnectionError(
                "Streaming connection failed"
            )
            
            with pytest.raises(Exception) as exc_info:
                custom_llm.call("Hello world")
            
            assert "Streaming connection failed" in str(exc_info.value)

    def test_non_custom_endpoint_error_handling(self):
        """Test that standard OpenAI endpoint errors are handled normally."""
        standard_llm = LLM(
            model="gpt-3.5-turbo",
            api_key="test-key"
        )
        
        captured_events = []
        
        def capture_event(sender, event):
            captured_events.append(event)
        
        with patch('crewai.utilities.events.crewai_event_bus.crewai_event_bus.emit', side_effect=capture_event):
            with patch('litellm.completion') as mock_completion:
                mock_completion.side_effect = Exception("Standard API error")
                
                with pytest.raises(Exception):
                    standard_llm.call("Hello world")
        
        assert len(captured_events) == 2  # Started and Failed events
        failed_event = captured_events[1]
        assert isinstance(failed_event, LLMCallFailedEvent)
        assert failed_event.error_type == "Exception"
        assert failed_event.original_error == "Standard API error"
        assert failed_event.endpoint_info is None  # No custom endpoint info

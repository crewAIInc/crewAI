import os
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from crewai.telemetry.telemetry import SafeBatchSpanProcessor, Telemetry


class TestTelemetry:
    def test_safe_batch_span_processor(self):
        """Test that SafeBatchSpanProcessor properly suppresses exceptions."""
        # Create a mock exporter that will be used by the processor
        mock_exporter = Mock()
        
        # Create a SafeBatchSpanProcessor with the mock exporter
        processor = SafeBatchSpanProcessor(mock_exporter)
        
        # Test force_flush with an exception
        with patch.object(BatchSpanProcessor, 'force_flush', side_effect=ConnectionError("Test error")):
            # This should not raise an exception
            processor.force_flush()
        
        # Test that the processor's export method suppresses exceptions
        with patch.object(mock_exporter, 'export', side_effect=ConnectionError("Test error")):
            # This should not raise an exception
            processor.export([])
    
    def test_telemetry_with_connection_error(self):
        """Test that telemetry connection errors are properly handled in real usage."""
        # Make sure telemetry is enabled for the test
        os.environ["OTEL_SDK_DISABLED"] = "false"
        
        # Create a telemetry instance
        telemetry = Telemetry()
        
        # Verify telemetry is initialized
        assert telemetry.ready is True
        
        # Test a real telemetry operation
        # This should not raise an exception even if there are connection issues
        telemetry.flow_creation_span("test_flow")
        
        # Reset environment variables
        os.environ["OTEL_SDK_DISABLED"] = "true"

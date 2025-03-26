import sys
import unittest
import threading
from unittest.mock import patch
from io import StringIO
import pytest

from crewai.utilities.logger import Logger


class TestLogger(unittest.TestCase):
    """Test suite for the Logger class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.logger = Logger(verbose=True)
        self.output = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.output

    def tearDown(self):
        """Clean up test environment after each test."""
        sys.stdout = self.old_stdout

    def test_log_in_sync_context(self):
        """Test logging in a regular synchronous context."""
        self.logger.log("info", "Test message")
        output = self.output.getvalue()
        self.assertIn("[INFO]: Test message", output)
        self.assertIn("\n", output)

    @patch('sys.stdout.flush')
    def test_stdout_is_flushed(self, mock_flush):
        """Test that stdout is properly flushed after writing."""
        self.logger.log("info", "Test message")
        mock_flush.assert_called_once()
    
    @pytest.mark.parametrize("log_level,message", [
        ("info", "Info message"),
        ("error", "Error message"),
        ("warning", "Warning message"),
        ("debug", "Debug message")
    ])
    def test_multiple_log_levels(self, log_level, message):
        """Test logging with different log levels."""
        self.logger.log(log_level, message)
        output = self.output.getvalue()
        self.assertIn(f"[{log_level.upper()}]: {message}", output)
    
    def test_thread_safety(self):
        """Test that logger is thread-safe."""
        messages = []
        for i in range(10):
            messages.append(f"Message {i}")
        
        threads = []
        for message in messages:
            thread = threading.Thread(
                target=lambda msg: self.logger.log("info", msg),
                args=(message,)
            )
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        output = self.output.getvalue()
        for message in messages:
            self.assertIn(message, output)


class TestFastAPICompatibility(unittest.TestCase):
    """Test compatibility with FastAPI."""
    
    def test_import_in_fastapi(self):
        """Test that logger can be imported in a FastAPI context."""
        try:
            import fastapi
            from crewai.utilities.logger import Logger
            logger = Logger(verbose=True)
            self.assertTrue(True)
        except ImportError:
            self.skipTest("FastAPI not installed")
        except Exception as e:
            self.fail(f"Unexpected error: {e}")

import sys
import unittest
from unittest.mock import patch
from io import StringIO

from crewai.utilities.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger(verbose=True)
        self.output = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.output

    def tearDown(self):
        sys.stdout = self.old_stdout

    def test_log_in_sync_context(self):
        self.logger.log("info", "Test message")
        output = self.output.getvalue()
        self.assertIn("[INFO]: Test message", output)
        self.assertIn("\n", output)

    @patch('sys.stdout.flush')
    def test_stdout_is_flushed(self, mock_flush):
        self.logger.log("info", "Test message")
        mock_flush.assert_called_once()


class TestFastAPICompatibility(unittest.TestCase):
    def test_import_in_fastapi(self):
        try:
            import fastapi
            from crewai.utilities.logger import Logger
            logger = Logger(verbose=True)
            self.assertTrue(True)
        except ImportError:
            self.skipTest("FastAPI not installed")
        except Exception as e:
            self.fail(f"Unexpected error: {e}")

import sys
import unittest
from unittest.mock import patch
import asyncio
import pytest
from io import StringIO

try:
    import fastapi
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    try:
        from httpx import AsyncClient
        ASYNC_CLIENT_AVAILABLE = True
    except ImportError:
        ASYNC_CLIENT_AVAILABLE = False
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    ASYNC_CLIENT_AVAILABLE = False

from crewai.utilities.logger import Logger


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not installed")
class TestFastAPILogger(unittest.TestCase):
    """Test suite for Logger class in FastAPI context."""
    
    def setUp(self):
        """Set up test environment before each test."""
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI not installed")
        
        self.app = FastAPI()
        self.logger = Logger(verbose=True)
        
        @self.app.get("/")
        async def root():
            self.logger.log("info", "This is a test log message from FastAPI")
            return {"message": "Hello World"}
        
        @self.app.get("/error")
        async def error_route():
            self.logger.log("error", "This is an error log message from FastAPI")
            return {"error": "Test error"}
        
        self.client = TestClient(self.app)
        
        self.output = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.output

    def tearDown(self):
        """Clean up test environment after each test."""
        sys.stdout = self.old_stdout

    def test_logger_in_fastapi_context(self):
        """Test that logger works in FastAPI context."""
        response = self.client.get("/")
        
        output = self.output.getvalue()
        self.assertIn("[INFO]: This is a test log message from FastAPI", output)
        self.assertIn("\n", output)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello World"})
    
    @pytest.mark.parametrize("route,log_level,expected_message", [
        ("/", "info", "This is a test log message from FastAPI"),
        ("/error", "error", "This is an error log message from FastAPI")
    ])
    def test_multiple_routes(self, route, log_level, expected_message):
        """Test logging from different routes with different log levels."""
        response = self.client.get(route)
        
        output = self.output.getvalue()
        self.assertIn(f"[{log_level.upper()}]: {expected_message}", output)
        self.assertEqual(response.status_code, 200)
    
    @unittest.skipIf(not ASYNC_CLIENT_AVAILABLE, "AsyncClient not available")
    @pytest.mark.asyncio
    async def test_async_logger_in_fastapi(self):
        """Test logger in async context using AsyncClient."""
        self.output = StringIO()
        sys.stdout = self.output
        
        async with AsyncClient(app=self.app, base_url="http://test") as ac:
            response = await ac.get("/")
            self.assertEqual(response.status_code, 200)
            
            output = self.output.getvalue()
            self.assertIn("[INFO]: This is a test log message from FastAPI", output)

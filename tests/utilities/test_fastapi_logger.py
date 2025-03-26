import sys
import unittest
from unittest.mock import patch
import asyncio
from io import StringIO

try:
    import fastapi
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from crewai.utilities.logger import Logger


@unittest.skipIf(not FASTAPI_AVAILABLE, "FastAPI not installed")
class TestFastAPILogger(unittest.TestCase):
    def setUp(self):
        if not FASTAPI_AVAILABLE:
            self.skipTest("FastAPI not installed")
        
        from fastapi import FastAPI
        
        self.app = FastAPI()
        self.logger = Logger(verbose=True)
        
        @self.app.get("/")
        async def root():
            self.logger.log("info", "This is a test log message from FastAPI")
            return {"message": "Hello World"}
        
        self.client = TestClient(self.app)
        
        self.output = StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.output

    def tearDown(self):
        sys.stdout = self.old_stdout

    def test_logger_in_fastapi_context(self):
        response = self.client.get("/")
        
        output = self.output.getvalue()
        self.assertIn("[INFO]: This is a test log message from FastAPI", output)
        self.assertIn("\n", output)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello World"})

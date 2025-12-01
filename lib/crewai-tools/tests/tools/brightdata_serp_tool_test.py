import unittest
from unittest.mock import MagicMock, patch

from crewai_tools.tools.brightdata_tool.brightdata_serp import BrightDataSearchTool


class TestBrightDataSearchTool(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
    )
    def setUp(self):
        self.tool = BrightDataSearchTool()

    @patch("requests.post")
    def test_run_successful_search(self, mock_post):
        # Sample mock JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock response text"
        mock_post.return_value = mock_response

        # Define search input
        input_data = {
            "query": "latest AI news",
            "search_engine": "google",
            "country": "us",
            "language": "en",
            "search_type": "nws",
            "device_type": "desktop",
            "parse_results": True,
            "save_file": False,
        }

        result = self.tool._run(**input_data)

        # Assertions
        self.assertIsInstance(result, str)  # Your tool returns response.text (string)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_run_with_request_exception(self, mock_post):
        mock_post.side_effect = Exception("Timeout")

        result = self.tool._run(query="AI", search_engine="google")
        self.assertIn("Error", result)

    def tearDown(self):
        # Clean up env vars
        pass


if __name__ == "__main__":
    unittest.main()

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

    @patch.dict(
        "os.environ",
        {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
    )
    def test_get_search_url_google_no_dollar_prefix(self):
        """Test that Google search URL does not contain a '$' prefix before the query."""
        tool = BrightDataSearchTool()
        url = tool.get_search_url("google", "AI news")
        self.assertEqual(url, "https://www.google.com/search?q=AI news")
        self.assertNotIn("$", url)

    @patch.dict(
        "os.environ",
        {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
    )
    def test_get_search_url_bing_no_dollar_prefix(self):
        """Test that Bing search URL does not contain a '$' prefix before the query."""
        tool = BrightDataSearchTool()
        url = tool.get_search_url("bing", "AI news")
        self.assertEqual(url, "https://www.bing.com/search?q=AI news")
        self.assertNotIn("$", url)

    @patch.dict(
        "os.environ",
        {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
    )
    def test_get_search_url_yandex_no_dollar_prefix(self):
        """Test that Yandex search URL does not contain a '$' prefix before the query."""
        tool = BrightDataSearchTool()
        url = tool.get_search_url("yandex", "AI news")
        self.assertEqual(url, "https://yandex.com/search/?text=AI news")
        self.assertNotIn("$", url)

    @patch.dict(
        "os.environ",
        {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
    )
    @patch("requests.post")
    def test_run_search_url_no_dollar_prefix(self, mock_post):
        """Test that the full _run flow produces URLs without '$' prefix in the query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock response text"
        mock_post.return_value = mock_response

        tool = BrightDataSearchTool()
        tool._run(query="test query", search_engine="google")

        call_args = mock_post.call_args
        request_body = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        url_sent = request_body["url"]
        self.assertNotIn("$", url_sent.split("?q=")[1])
        self.assertIn("q=test%20query", url_sent)

    def tearDown(self):
        # Clean up env vars
        pass


if __name__ == "__main__":
    unittest.main()

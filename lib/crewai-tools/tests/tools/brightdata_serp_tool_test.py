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
    @patch("requests.post")
    def test_run_uses_constructor_options(self, mock_post):
        mock_post.return_value = MagicMock(text="mock response text")
        tool = BrightDataSearchTool(
            search_engine="bing", country="gb", language="fr", device_type="mobile"
        )

        tool.run(query="ai news")

        url = mock_post.call_args.kwargs["json"]["url"]
        self.assertTrue(url.startswith("https://www.bing.com/search?q=ai%20news"))
        self.assertIn("gl=gb", url)
        self.assertIn("hl=fr", url)
        self.assertIn("brd_mobile=1", url)

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()

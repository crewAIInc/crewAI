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

    def test_get_search_url_google(self):
        url = self.tool.get_search_url("google", "hello+world")
        self.assertIn("google.com/search", url)
        self.assertIn("q=hello+world", url)
        self.assertNotIn("$", url)

    def test_get_search_url_bing(self):
        url = self.tool.get_search_url("bing", "hello+world")
        self.assertIn("bing.com/search", url)
        self.assertIn("q=hello+world", url)
        self.assertNotIn("$", url)

    def test_get_search_url_yandex(self):
        url = self.tool.get_search_url("yandex", "hello+world")
        self.assertIn("yandex.com/search", url)
        self.assertIn("text=hello+world", url)
        self.assertNotIn("$", url)

    def test_get_search_url_default_is_google(self):
        url = self.tool.get_search_url("unknown_engine", "query")
        self.assertIn("google.com", url)
        self.assertNotIn("$", url)

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
        }

        result = self.tool._run(**input_data)

        # Assertions
        self.assertIsInstance(result, str)
        mock_post.assert_called_once()

        # Verify the request URL does not contain '$'
        call_args = mock_post.call_args
        request_params = call_args[1]["json"] if call_args[1] else call_args[0][1]
        url_in_request = request_params.get("url", "")
        self.assertNotIn("$", url_in_request)

    @patch("requests.post")
    def test_run_with_custom_results_count(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock response text"
        mock_post.return_value = mock_response

        self.tool._run(query="AI news", results_count=5)

        call_args = mock_post.call_args
        request_params = call_args[1]["json"] if call_args[1] else call_args[0][1]
        url_in_request = request_params.get("url", "")
        self.assertIn("num=5", url_in_request)

    @patch("requests.post")
    def test_run_default_results_count_is_10(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "mock response text"
        mock_post.return_value = mock_response

        self.tool._run(query="AI news")

        call_args = mock_post.call_args
        request_params = call_args[1]["json"] if call_args[1] else call_args[0][1]
        url_in_request = request_params.get("url", "")
        self.assertIn("num=10", url_in_request)

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

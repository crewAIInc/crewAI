from unittest.mock import Mock, patch

from crewai_tools.tools.brightdata_tool.brightdata_unlocker import (
    BrightDataWebUnlockerTool,
)
import requests


@patch.dict(
    "os.environ",
    {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
)
@patch("crewai_tools.tools.brightdata_tool.brightdata_unlocker.requests.post")
def test_run_success_html(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    tool = BrightDataWebUnlockerTool()
    tool._run(url="https://example.com", format="html", save_file=False)


@patch.dict(
    "os.environ",
    {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
)
@patch("crewai_tools.tools.brightdata_tool.brightdata_unlocker.requests.post")
def test_run_success_json(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "mock response text"
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    tool = BrightDataWebUnlockerTool()
    result = tool._run(url="https://example.com", format="json")

    assert isinstance(result, str)


@patch.dict(
    "os.environ",
    {"BRIGHT_DATA_API_KEY": "test_api_key", "BRIGHT_DATA_ZONE": "test_zone"},
)
@patch("crewai_tools.tools.brightdata_tool.brightdata_unlocker.requests.post")
def test_run_http_error(mock_post):
    mock_response = Mock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        response=mock_response
    )
    mock_post.return_value = mock_response

    tool = BrightDataWebUnlockerTool()
    result = tool._run(url="https://example.com")

    assert "HTTP Error" in result
    assert "Forbidden" in result

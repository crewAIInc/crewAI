from unittest.mock import patch, MagicMock
from crewai_tools import HackerNewsTopStoriesTool

@patch("requests.get")
def test_hacker_news_tool_success(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = [99999]

    mock_item = MagicMock(status_code=200)
    mock_item.json.return_value = {
        "title": "Sample Tech Announcement",
        "url": "https://example.com/announcement",
        "score": 250
    }

    mock_get.side_effect = [mock_top, mock_item]

    tool = HackerNewsTopStoriesTool()
    output = tool._run(limit=1)

    assert "Sample Tech Announcement" in output
    assert "https://example.com/announcement" in output

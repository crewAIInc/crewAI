from unittest.mock import MagicMock, patch

import requests

from crewai_tools import HackerNewsTopStoriesTool


@patch("requests.get")
def test_hacker_news_tool_success(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = [99999]

    mock_item = MagicMock(status_code=200)
    mock_item.json.return_value = {
        "type": "story",
        "title": "Sample Tech Announcement",
        "url": "https://example.com/announcement",
        "score": 250,
    }

    mock_get.side_effect = [mock_top, mock_item]

    tool = HackerNewsTopStoriesTool()
    output = tool._run(limit=1)

    assert "Sample Tech Announcement" in output
    assert "https://example.com/announcement" in output


@patch("requests.get")
def test_hacker_news_tool_skips_null_item(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = [99999]

    mock_item = MagicMock(status_code=200)
    mock_item.json.return_value = None

    mock_get.side_effect = [mock_top, mock_item]

    tool = HackerNewsTopStoriesTool()

    assert tool._run(limit=1) == "No stories retrieved."


@patch("requests.get")
def test_hacker_news_tool_skips_non_story_and_continues(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = [1, 2, 3]

    mock_job = MagicMock(status_code=200)
    mock_job.json.return_value = {
        "type": "job",
        "title": "Some Job",
    }

    mock_story_1 = MagicMock(status_code=200)
    mock_story_1.json.return_value = {
        "type": "story",
        "title": "Story One",
        "url": "https://example.com/one",
        "score": 100,
    }

    mock_story_2 = MagicMock(status_code=200)
    mock_story_2.json.return_value = {
        "type": "story",
        "title": "Story Two",
        "url": "https://example.com/two",
        "score": 200,
    }

    mock_get.side_effect = [
        mock_top,
        mock_job,
        mock_story_1,
        mock_story_2,
    ]

    tool = HackerNewsTopStoriesTool()
    output = tool._run(limit=2)

    assert "Story One" in output
    assert "Story Two" in output
    assert "Some Job" not in output
    assert "1. Story One" in output
    assert "2. Story Two" in output
    assert mock_get.call_count == 4


@patch("requests.get")
def test_hacker_news_tool_skips_deleted_item(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = [99999]

    mock_item = MagicMock(status_code=200)
    mock_item.json.return_value = {
        "id": 99999,
        "type": "story",
        "deleted": True,
    }

    mock_get.side_effect = [mock_top, mock_item]

    tool = HackerNewsTopStoriesTool()

    assert tool._run(limit=1) == "No stories retrieved."


@patch("requests.get")
def test_hacker_news_tool_continues_after_item_request_failure(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = [1, 2]

    mock_story = MagicMock(status_code=200)
    mock_story.json.return_value = {
        "type": "story",
        "title": "Story After Failure",
        "url": "https://example.com/story",
        "score": 100,
    }

    mock_get.side_effect = [
        mock_top,
        requests.Timeout("request timed out"),
        mock_story,
    ]

    tool = HackerNewsTopStoriesTool()
    output = tool._run(limit=1)

    assert "Story After Failure" in output
    assert mock_get.call_count == 3


@patch("requests.get")
def test_hacker_news_tool_bounds_item_lookups(mock_get):
    mock_top = MagicMock(status_code=200)
    mock_top.json.return_value = list(range(1, 101))

    mock_item = MagicMock(status_code=200)
    mock_item.json.return_value = None

    mock_get.side_effect = [mock_top] + [mock_item] * 20

    tool = HackerNewsTopStoriesTool()
    output = tool._run(limit=1)

    assert output == "No stories retrieved."
    assert mock_get.call_count == 21

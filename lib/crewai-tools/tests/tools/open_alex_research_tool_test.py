from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools import OpenAlexResearchTool
from crewai_tools.tools.open_alex_research_tool.open_alex_research_tool import (
    OpenAlexResearchToolInput,
)


@patch("requests.get")
def test_open_alex_research_tool_success(mock_get):
    """Return formatted scholarly results for a successful search."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Attention Is All You Need",
                "publication_year": 2017,
                "cited_by_count": 95000,
                "primary_location": {"source": {"display_name": "NeurIPS"}},
                "authorships": [
                    {"author": {"display_name": "Ashish Vaswani"}},
                    {"author": {"display_name": "Noam Shazeer"}},
                ],
                "open_access": {"oa_url": "https://arxiv.org/abs/1706.03762"},
                "abstract_inverted_index": {
                    "The": [0],
                    "dominant": [1],
                    "architecture": [2],
                },
            }
        ]
    }

    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    result = tool._run(
        query="Attention Is All You Need",
        limit=1,
    )

    assert "Attention Is All You Need" in result
    assert "Ashish Vaswani" in result
    assert "95000" in result
    assert "NeurIPS" in result
    assert "https://arxiv.org/abs/1706.03762" in result
    assert "The dominant architecture" in result


@patch("requests.get")
def test_open_alex_research_tool_no_results(mock_get):
    """Return a no-results message when OpenAlex returns no works."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"results": []}

    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    result = tool._run(query="NonExistentResearchTopic12345")

    assert "No scholarly works found" in result


@patch("requests.get")
def test_open_alex_research_tool_empty_query(mock_get):
    """Reject an empty search query without making an HTTP request."""
    tool = OpenAlexResearchTool()

    result = tool._run(query="   ")

    assert result == "Error: Search query must be a non-empty string."
    mock_get.assert_not_called()


@patch("requests.get")
def test_open_alex_research_tool_request_error(mock_get):
    """Return an error message when the OpenAlex request fails."""
    mock_get.side_effect = requests.RequestException("OpenAlex unavailable")

    tool = OpenAlexResearchTool()
    result = tool._run(query="machine learning")

    assert "Error retrieving OpenAlex research data" in result
    assert "OpenAlex unavailable" in result


@patch("requests.get")
def test_open_alex_research_tool_uses_query_parameters(mock_get):
    """Pass the search query and result limit as HTTP parameters."""    
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"results": []}

    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    tool._run(query="machine learning & security", limit=7)

    _, kwargs = mock_get.call_args

    assert kwargs["params"]["search"] == "machine learning & security"
    assert kwargs["params"]["per_page"] == 7
    assert kwargs["timeout"] == 10

@patch("requests.get")
def test_open_alex_research_tool_uses_email_parameter(mock_get):
    """Pass an optional email address as OpenAlex's mailto parameter."""
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"results": []}

    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    tool._run(
        query="machine learning",
        limit=5,
        email="researcher@example.com",
    )

    _, kwargs = mock_get.call_args

    assert kwargs["params"]["search"] == "machine learning"
    assert kwargs["params"]["per_page"] == 5
    assert kwargs["params"]["mailto"] == "researcher@example.com"

def test_open_alex_research_tool_reconstructs_abstract():
    """Reconstruct abstract text from OpenAlex's inverted index."""
    tool = OpenAlexResearchTool()

    result = tool._reconstruct_abstract(
        {
            "Retrieval": [1],
            "augmented": [2],
            "generation": [0],
        }
    )

    assert result == "generation Retrieval augmented"


def test_open_alex_research_tool_rejects_limit_below_minimum():
    """Reject result limits below the supported minimum."""
    with pytest.raises(ValueError):
        OpenAlexResearchToolInput(
            query="machine learning",
            limit=0,
        )


def test_open_alex_research_tool_rejects_limit_above_maximum():
    """Reject result limits above the supported maximum."""
    with pytest.raises(ValueError):
        OpenAlexResearchToolInput(
            query="machine learning",
            limit=21,
        )

def test_open_alex_research_tool_rejects_positional_limit_above_maximum():
    """Reject an out-of-range limit passed positionally to run()."""
    tool = OpenAlexResearchTool()

    with pytest.raises(ValueError):
        tool.run("machine learning", 21)

@patch("requests.get")
def test_open_alex_research_tool_http_error(mock_get):
    """Return an error message when OpenAlex returns an HTTP error."""
    mock_response = MagicMock(status_code=500)
    mock_response.raise_for_status.side_effect = requests.HTTPError("HTTP 500")
    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    result = tool._run(query="Attention Is All You Need")

    assert result == "Error retrieving OpenAlex research data: HTTP 500"

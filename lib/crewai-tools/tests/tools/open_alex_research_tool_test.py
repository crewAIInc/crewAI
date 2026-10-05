from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools import OpenAlexResearchTool
from crewai_tools.tools.open_alex_research_tool.open_alex_research_tool import (
    OpenAlexResearchToolInput,
)


@patch("requests.get")
def test_open_alex_research_tool_success(mock_get):
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
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"results": []}

    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    result = tool._run(query="NonExistentResearchTopic12345")

    assert "No scholarly works found" in result


@patch("requests.get")
def test_open_alex_research_tool_empty_query(mock_get):
    tool = OpenAlexResearchTool()

    result = tool._run(query="   ")

    assert result == "Error: Search query must be a non-empty string."
    mock_get.assert_not_called()


@patch("requests.get")
def test_open_alex_research_tool_request_error(mock_get):
    mock_get.side_effect = requests.RequestException("OpenAlex unavailable")

    tool = OpenAlexResearchTool()
    result = tool._run(query="machine learning")

    assert "Error retrieving OpenAlex research data" in result
    assert "OpenAlex unavailable" in result


@patch("requests.get")
def test_open_alex_research_tool_uses_query_parameters(mock_get):
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"results": []}

    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    tool._run(query="machine learning & security", limit=7)

    _, kwargs = mock_get.call_args

    assert kwargs["params"]["search"] == "machine learning & security"
    assert kwargs["params"]["per_page"] == 7
    assert kwargs["timeout"] == 10


def test_open_alex_research_tool_reconstructs_abstract():
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
    with pytest.raises(ValueError):
        OpenAlexResearchToolInput(
            query="machine learning",
            limit=0,
        )


def test_open_alex_research_tool_rejects_limit_above_maximum():
    with pytest.raises(ValueError):
        OpenAlexResearchToolInput(
            query="machine learning",
            limit=21,
        )

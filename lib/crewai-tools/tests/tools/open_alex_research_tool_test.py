from unittest.mock import MagicMock, patch
from crewai_tools.tools.open_alex_research_tool.open_alex_research_tool import OpenAlexResearchTool


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
                "abstract_inverted_index": {"The": [0], "dominant": [1]},
            }
        ]
    }
    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    result = tool._run(query="Attention Is All You Need", limit=1)

    assert "Attention Is All You Need" in result
    assert "Ashish Vaswani" in result
    assert "95000" in result
    assert "NeurIPS" in result


@patch("requests.get")
def test_open_alex_research_tool_no_results(mock_get):
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"results": []}
    mock_get.return_value = mock_response

    tool = OpenAlexResearchTool()
    result = tool._run(query="NonExistentResearchTopic12345")

    assert "No scholarly works found" in result

from unittest.mock import MagicMock, patch

from crewai_tools import BGPTPaperTool


@patch("crewai_tools.tools.bgpt_tool.bgpt_tool.requests.post")
def test_search_papers(mock_post: MagicMock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "results": [
                {
                    "title": "Example study",
                    "doi": "10.1000/example",
                    "paper_limitations_and_biases": "Small sample",
                }
            ]
        },
    )
    tool = BGPTPaperTool()
    result = tool.run(search_query="CRISPR", num_results=1)
    assert "Example study" in result
    assert "paper_limitations_and_biases" in result
    mock_post.assert_called_once()


@patch("crewai_tools.tools.bgpt_tool.bgpt_tool.requests.post")
def test_lookup_paper(mock_post: MagicMock) -> None:
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "result": {
                "title": "DOI paper",
                "doi": "10.1000/doi",
                "how_to_falsify": "Repeat with larger cohort",
            }
        },
    )
    tool = BGPTPaperTool()
    result = tool.run(doi="10.1000/doi")
    assert "how_to_falsify" in result

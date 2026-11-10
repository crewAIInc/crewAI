from types import SimpleNamespace
from typing import Any

from crewai_tools.tools.linkup.linkup_search_tool import LinkupSearchTool


class FakeLinkupClient:
    def __init__(self, response: Any) -> None:
        self.response = response

    def search(self, query: str, depth: str, output_type: str) -> Any:
        return self.response


def make_tool(response: Any) -> LinkupSearchTool:
    tool = LinkupSearchTool(api_key="test-key")
    tool._client = FakeLinkupClient(response)
    return tool


def test_search_results_output_is_unchanged() -> None:
    tool = make_tool(
        SimpleNamespace(
            results=[
                SimpleNamespace(name="Example", url="https://example.com", content="c")
            ]
        )
    )

    assert tool._run(query="question") == {
        "success": True,
        "results": [
            {"name": "Example", "url": "https://example.com", "content": "c"}
        ],
    }


def test_sourced_answer_output_returns_answer_and_sources() -> None:
    tool = make_tool(
        SimpleNamespace(
            answer="The answer",
            sources=[
                SimpleNamespace(
                    name="Example", url="https://example.com", snippet="snippet"
                )
            ],
        )
    )

    assert tool._run(query="question", output_type="sourcedAnswer") == {
        "success": True,
        "answer": "The answer",
        "sources": [
            {"name": "Example", "url": "https://example.com", "snippet": "snippet"}
        ],
    }

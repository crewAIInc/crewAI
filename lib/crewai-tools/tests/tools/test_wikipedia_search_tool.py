from types import SimpleNamespace
from unittest.mock import patch

from crewai_tools import WikipediaSearchTool


class FakeDisambiguationError(Exception):
    def __init__(self, title: str, options: list[str]) -> None:
        super().__init__(title)
        self.title = title
        self.options = options


class FakePageError(Exception):
    pass


def test_wikipedia_search_tool_success():
    tool = WikipediaSearchTool()

    fake_wikipedia = SimpleNamespace(
        summary=lambda query, sentences, auto_suggest: (
            "Alan Turing was a mathematician. "
            "He was a pioneer of computer science."
        ),
        exceptions=SimpleNamespace(
            DisambiguationError=FakeDisambiguationError,
            PageError=FakePageError,
        ),
    )

    with patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WIKIPEDIA_AVAILABLE",
        True,
    ), patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.wikipedia",
        fake_wikipedia,
    ):
        result = tool._run("Alan Turing")

    assert "Alan Turing" in result


def test_wikipedia_search_tool_disambiguation():
    tool = WikipediaSearchTool()

    def raise_disambiguation(
        query: str, sentences: int, auto_suggest: bool
    ) -> str:
        raise FakeDisambiguationError(
            "Mercury",
            ["Mercury (planet)", "Mercury (element)", "Mercury (mythology)"],
        )

    fake_wikipedia = SimpleNamespace(
        summary=raise_disambiguation,
        exceptions=SimpleNamespace(
            DisambiguationError=FakeDisambiguationError,
            PageError=FakePageError,
        ),
    )

    with patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WIKIPEDIA_AVAILABLE",
        True,
    ), patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.wikipedia",
        fake_wikipedia,
    ):
        result = tool._run("Mercury")

    assert "ambiguous" in result.lower()
    assert "Mercury (planet)" in result


def test_wikipedia_search_tool_page_error():
    tool = WikipediaSearchTool()

    def raise_page_error(query: str, sentences: int, auto_suggest: bool) -> str:
        raise FakePageError("UnknownTopic")

    fake_wikipedia = SimpleNamespace(
        summary=raise_page_error,
        exceptions=SimpleNamespace(
            DisambiguationError=FakeDisambiguationError,
            PageError=FakePageError,
        ),
    )

    with patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WIKIPEDIA_AVAILABLE",
        True,
    ), patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.wikipedia",
        fake_wikipedia,
    ):
        result = tool._run("UnknownTopic")

    assert "no wikipedia page" in result.lower()


def test_wikipedia_search_tool_unexpected_error():
    tool = WikipediaSearchTool()

    def raise_unexpected(query: str, sentences: int, auto_suggest: bool) -> str:
        raise Exception("API failure")

    fake_wikipedia = SimpleNamespace(
        summary=raise_unexpected,
        exceptions=SimpleNamespace(
            DisambiguationError=FakeDisambiguationError,
            PageError=FakePageError,
        ),
    )

    with patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WIKIPEDIA_AVAILABLE",
        True,
    ), patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.wikipedia",
        fake_wikipedia,
    ):
        result = tool._run("Anything")

    assert "an error occurred while searching wikipedia" in result.lower()
    assert "API failure" in result


def test_wikipedia_search_tool_missing_dependency():
    tool = WikipediaSearchTool()

    with patch(
        "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WIKIPEDIA_AVAILABLE",
        False,
    ):
        try:
            tool._run("Anything")
            assert False, "Expected ImportError"
        except ImportError as e:
            assert "wikipedia" in str(e).lower()

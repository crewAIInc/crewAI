from unittest.mock import MagicMock, patch

import pytest

from crewai_tools import WikipediaSearchTool

MODULE_PATH = "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool"


@pytest.fixture
def tool():
    return WikipediaSearchTool()


@pytest.fixture
def tool_short():
    return WikipediaSearchTool(sentences=2)


@pytest.fixture
def tool_multi():
    return WikipediaSearchTool(top_k=3)


def _make_mock_wikipedia():
    mock_wiki = MagicMock()
    mock_wiki.exceptions = MagicMock()
    mock_wiki.exceptions.DisambiguationError = type(
        "DisambiguationError", (Exception,), {}
    )
    mock_wiki.exceptions.PageError = type("PageError", (Exception,), {})
    return mock_wiki


class FakePage:
    def __init__(self, title, url="https://en.wikipedia.org/wiki/Test"):
        self.title = title
        self.url = url


class TestWikipediaSearchToolInit:
    def test_default_params(self, tool):
        assert tool.sentences == 5
        assert tool.language == "en"
        assert tool.top_k == 1

    def test_custom_params(self):
        tool = WikipediaSearchTool(sentences=3, language="fr", top_k=2)
        assert tool.sentences == 3
        assert tool.language == "fr"
        assert tool.top_k == 2

    def test_tool_metadata(self, tool):
        assert tool.name == "Wikipedia Search"
        assert "Wikipedia" in tool.description


class TestWikipediaSearchToolRun:
    def test_basic_search(self, tool):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["Python (programming language)"]
        mock_wiki.summary.return_value = "Python is a programming language."
        mock_wiki.page.return_value = FakePage(
            "Python (programming language)",
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
        )

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="Python programming")

        mock_wiki.set_lang.assert_called_once_with("en")
        mock_wiki.search.assert_called_once_with("Python programming", results=1)
        assert "Python (programming language)" in result
        assert "Python is a programming language." in result
        assert "https://en.wikipedia.org/wiki/Python_(programming_language)" in result

    def test_no_results(self, tool):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = []

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="xyznonexistenttopic123")

        assert "No Wikipedia articles found" in result

    def test_disambiguation_error(self, tool):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["Mercury"]

        exc = mock_wiki.exceptions.DisambiguationError(
            "Mercury", ["Mercury (planet)", "Mercury (element)", "Freddie Mercury"]
        )
        exc.options = ["Mercury (planet)", "Mercury (element)", "Freddie Mercury"]
        mock_wiki.summary.side_effect = exc

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="Mercury")

        assert "Disambiguation" in result
        assert "Mercury (planet)" in result

    def test_page_not_found(self, tool):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["NonexistentPage12345"]
        mock_wiki.summary.side_effect = mock_wiki.exceptions.PageError(
            "NonexistentPage12345"
        )

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="NonexistentPage12345")

        assert "No Wikipedia page found" in result

    def test_generic_exception(self, tool):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["Test"]
        mock_wiki.summary.side_effect = RuntimeError("Network error")

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="Test")

        assert "Failed to retrieve page" in result

    def test_search_error(self, tool):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.side_effect = Exception("API error")

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="Test")

        assert "Error searching Wikipedia" in result

    def test_custom_sentences(self, tool_short):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["Test"]
        mock_wiki.summary.return_value = "Short summary."
        mock_wiki.page.return_value = FakePage("Test")

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            tool_short._run(search_query="Test")

        mock_wiki.summary.assert_called_once_with("Test", sentences=2)

    def test_multiple_results(self, tool_multi):
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["AI", "ML", "DL"]
        mock_wiki.summary.side_effect = [
            "AI summary.",
            "ML summary.",
            "DL summary.",
        ]
        mock_wiki.page.side_effect = [
            FakePage("Artificial intelligence"),
            FakePage("Machine learning"),
            FakePage("Deep learning"),
        ]

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool_multi._run(search_query="AI")

        mock_wiki.search.assert_called_once_with("AI", results=3)
        assert "AI summary." in result
        assert "ML summary." in result
        assert "DL summary." in result
        assert result.count("---") == 2

    def test_language_setting(self):
        tool = WikipediaSearchTool(language="fr")
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["Test"]
        mock_wiki.summary.return_value = "Résumé du test."
        mock_wiki.page.return_value = FakePage("Test")

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            tool._run(search_query="Test")

        mock_wiki.set_lang.assert_called_once_with("fr")

    def test_missing_wikipedia_package(self, tool):
        with patch(f"{MODULE_PATH}.wikipedia", None):
            result = tool._run(search_query="Test")

        assert "wikipedia" in result.lower()
        assert "pip install" in result


class TestWikipediaSearchToolMixedResults:
    def test_mixed_success_and_disambiguation(self):
        tool = WikipediaSearchTool(top_k=2)
        mock_wiki = _make_mock_wikipedia()
        mock_wiki.search.return_value = ["Python", "Mercury"]

        exc = mock_wiki.exceptions.DisambiguationError(
            "Mercury", ["Mercury (planet)", "Mercury (element)"]
        )
        exc.options = ["Mercury (planet)", "Mercury (element)"]

        mock_wiki.summary.side_effect = [
            "Python is a language.",
            exc,
        ]
        mock_wiki.page.return_value = FakePage("Python")

        with patch(f"{MODULE_PATH}.wikipedia", mock_wiki):
            result = tool._run(search_query="Python Mercury")

        assert "Python is a language." in result
        assert "Disambiguation" in result

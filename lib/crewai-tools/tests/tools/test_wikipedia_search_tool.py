from unittest.mock import MagicMock, patch

from crewai_tools import WikipediaSearchTool
import pytest


@pytest.fixture
def tool():
    return WikipediaSearchTool()


@patch("wikipedia.summary")
@patch("wikipedia.page")
@patch("wikipedia.search")
def test_wikipedia_search_success(mock_search, mock_page, mock_summary, tool):
    mock_search.return_value = ["Python (programming language)"]

    mock_page_obj = MagicMock()
    mock_page_obj.title = "Python (programming language)"
    mock_page_obj.url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    mock_page.return_value = mock_page_obj
    mock_summary.return_value = "Python is a high-level programming language."

    result = tool._run(search_query="Python")

    assert "Title: Python (programming language)" in result
    assert "URL: https://en.wikipedia.org/wiki/Python_(programming_language)" in result
    assert "Summary: Python is a high-level programming language." in result
    mock_search.assert_called_once_with("Python", results=3)


@patch("wikipedia.page")
@patch("wikipedia.search")
def test_wikipedia_search_load_full_content(mock_search, mock_page, tool):
    mock_search.return_value = ["Python (programming language)"]

    mock_page_obj = MagicMock()
    mock_page_obj.title = "Python (programming language)"
    mock_page_obj.url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    mock_page_obj.content = "Full article content of Python programming language..."
    mock_page.return_value = mock_page_obj

    result = tool._run(search_query="Python", load_full_content=True)

    assert "Title: Python (programming language)" in result
    assert "Content: Full article content of Python programming language..." in result


@patch("wikipedia.search")
def test_wikipedia_search_no_results(mock_search, tool):
    mock_search.return_value = []

    result = tool._run(search_query="NonExistentTopic12345")

    assert "No Wikipedia results found for query: 'NonExistentTopic12345'" in result


@patch("wikipedia.page")
@patch("wikipedia.search")
def test_wikipedia_search_disambiguation(mock_search, mock_page, tool):
    import wikipedia

    mock_search.return_value = ["Python"]
    mock_page.side_effect = wikipedia.exceptions.DisambiguationError(
        "Python", ["Pythonidae", "Monty Python"]
    )

    result = tool._run(search_query="Python")

    assert "Disambiguation" in result
    assert "Pythonidae, Monty Python" in result


@patch("wikipedia.page")
@patch("wikipedia.search")
def test_wikipedia_search_page_error(mock_search, mock_page, tool):
    import wikipedia

    mock_search.return_value = ["MissingPage"]
    mock_page.side_effect = wikipedia.exceptions.PageError("MissingPage")

    result = tool._run(search_query="MissingPage")

    assert "Could not retrieve page details" in result


@patch("wikipedia.set_lang")
@patch("wikipedia.summary")
@patch("wikipedia.page")
@patch("wikipedia.search")
def test_wikipedia_search_custom_lang_and_limit(
    mock_search, mock_page, mock_summary, mock_set_lang, tool
):
    mock_search.return_value = ["Yapay zeka"]
    mock_page_obj = MagicMock()
    mock_page_obj.title = "Yapay zeka"
    mock_page_obj.url = "https://tr.wikipedia.org/wiki/Yapay_zeka"
    mock_page.return_value = mock_page_obj
    mock_summary.return_value = "Yapay zeka, bilgisayar biliminin bir dalıdır."

    result = tool._run(search_query="Yapay zeka", lang="tr", limit=1)

    assert "Title: Yapay zeka" in result
    mock_set_lang.assert_called_with("tr")
    mock_search.assert_called_once_with("Yapay zeka", results=1)


@patch("wikipedia.search")
def test_wikipedia_search_exception(mock_search, tool):
    mock_search.side_effect = Exception("API Connection Error")

    result = tool._run(search_query="Python")

    assert "Error searching Wikipedia for 'Python': API Connection Error" in result

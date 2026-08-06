from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from crewai_tools import WikipediaSearchTool
from crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool import (
    DisambiguationError,
    PageError,
    WikipediaClient,
)


@pytest.fixture
def tool():
    return WikipediaSearchTool()


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_success(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Python (programming language)"]

    mock_page_obj = MagicMock()
    mock_page_obj.title = "Python (programming language)"
    mock_page_obj.url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    mock_client.page.return_value = mock_page_obj
    mock_client.summary.return_value = "Python is a high-level programming language."

    result = tool._run(search_query="Python")

    assert "Title: Python (programming language)" in result
    assert "URL: https://en.wikipedia.org/wiki/Python_(programming_language)" in result
    assert "Summary: Python is a high-level programming language." in result
    mock_client.search.assert_called_once_with("Python", results=3)
    mock_client_cls.assert_called_once_with(lang="en", user_agent=tool.user_agent)


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_load_full_content(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Python (programming language)"]

    mock_page_obj = MagicMock()
    mock_page_obj.title = "Python (programming language)"
    mock_page_obj.url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
    mock_page_obj.content = (
        "Full article content of Python programming language..."
    )
    mock_client.page.return_value = mock_page_obj

    result = tool._run(search_query="Python", load_full_content=True)

    assert "Title: Python (programming language)" in result
    assert (
        "Content: Full article content of Python programming language..." in result
    )


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_no_results(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = []

    result = tool._run(search_query="NonExistentTopic12345")

    assert "No Wikipedia results found for query: 'NonExistentTopic12345'" in result


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_disambiguation(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Python"]
    mock_client.page.side_effect = DisambiguationError(
        "Python", ["Pythonidae", "Monty Python"]
    )

    result = tool._run(search_query="Python")

    assert "Disambiguation" in result
    assert "Pythonidae, Monty Python" in result


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_page_error(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["MissingPage"]
    mock_client.page.side_effect = PageError("MissingPage")

    result = tool._run(search_query="MissingPage")

    assert "Could not retrieve page details" in result


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_custom_lang_and_limit(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Yapay zeka"]
    mock_page_obj = MagicMock()
    mock_page_obj.title = "Yapay zeka"
    mock_page_obj.url = "https://tr.wikipedia.org/wiki/Yapay_zeka"
    mock_client.page.return_value = mock_page_obj
    mock_client.summary.return_value = "Yapay zeka, bilgisayar biliminin bir dalıdır."

    result = tool._run(search_query="Yapay zeka", lang="tr", limit=1)

    assert "Title: Yapay zeka" in result
    mock_client_cls.assert_called_once_with(lang="tr", user_agent=tool.user_agent)
    mock_client.search.assert_called_once_with("Yapay zeka", results=1)


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_wikipedia_search_exception(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = Exception("API Connection Error")

    result = tool._run(search_query="Python")

    assert "Error searching Wikipedia for 'Python': API Connection Error" in result


def test_wikipedia_client_instance_isolation():
    client1 = WikipediaClient(lang="en", user_agent="Agent-1")
    client2 = WikipediaClient(lang="tr", user_agent="Agent-2")

    assert client1.lang == "en"
    assert client1.api_url == "https://en.wikipedia.org/w/api.php"
    assert client1.user_agent == "Agent-1"

    assert client2.lang == "tr"
    assert client2.api_url == "https://tr.wikipedia.org/w/api.php"
    assert client2.user_agent == "Agent-2"


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_limit_validation_runtime_clamping_zero(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Topic"]

    tool._run(search_query="Test", limit=0)
    mock_client.search.assert_called_once_with("Test", results=1)


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_limit_validation_runtime_clamping_above_max(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Topic"]

    tool._run(search_query="Test", limit=15)
    mock_client.search.assert_called_once_with("Test", results=10)


def test_limit_validation_schema():
    with pytest.raises(ValidationError):
        WikipediaSearchTool(limit=0)

    with pytest.raises(ValidationError):
        WikipediaSearchTool(limit=11)


@patch(
    "crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient"
)
def test_string_joining_with_separator_multiple_results(mock_client_cls, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.return_value = ["Python", "Java"]

    page1 = MagicMock()
    page1.title = "Python"
    page1.url = "https://en.wikipedia.org/wiki/Python"

    page2 = MagicMock()
    page2.title = "Java"
    page2.url = "https://en.wikipedia.org/wiki/Java"

    mock_client.page.side_effect = [page1, page2]
    mock_client.summary.side_effect = [
        "Python summary text.",
        "Java summary text.",
    ]

    result = tool._run(search_query="Programming", limit=2)

    separator = "\n\n" + "-" * 80 + "\n\n"

    assert separator in result
    assert not result.startswith(separator)
    assert not result.endswith(separator)

    res1_part = "Title: Python\nURL: https://en.wikipedia.org/wiki/Python\nSummary: Python summary text."
    res2_part = "Title: Java\nURL: https://en.wikipedia.org/wiki/Java\nSummary: Java summary text."
    assert result == f"{res1_part}{separator}{res2_part}"


@patch("requests.get")
def test_wikipedia_client_search_http_mock(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "query": {
            "search": [
                {"title": "Python (programming language)"},
                {"title": "Python (snake)"},
            ]
        }
    }
    mock_get.return_value = mock_response

    client = WikipediaClient(lang="en", user_agent="TestAgent/1.0")
    titles = client.search("Python", results=2)

    assert titles == ["Python (programming language)", "Python (snake)"]
    mock_get.assert_called_once_with(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": "Python",
            "srlimit": 2,
            "format": "json",
        },
        headers={"User-Agent": "TestAgent/1.0"},
        timeout=10,
    )


@patch("requests.get")
def test_wikipedia_client_page_and_summary_http_mock(mock_get):
    mock_page_res = MagicMock()
    mock_page_res.json.return_value = {
        "query": {
            "pages": {
                "12345": {
                    "pageid": 12345,
                    "title": "Python",
                    "fullurl": "https://en.wikipedia.org/wiki/Python",
                }
            }
        }
    }
    mock_summary_res = MagicMock()
    mock_summary_res.json.return_value = {
        "query": {
            "pages": {
                "12345": {"extract": "Python is an interpreted programming language."}
            }
        }
    }
    mock_get.side_effect = [mock_page_res, mock_summary_res]

    client = WikipediaClient(lang="en", user_agent="TestAgent/1.0")
    page = client.page("Python")

    assert page.title == "Python"
    assert page.url == "https://en.wikipedia.org/wiki/Python"

    summary = client.summary("Python")
    assert summary == "Python is an interpreted programming language."


@patch("requests.get")
def test_wikipedia_client_get_content_http_mock(mock_get):
    mock_res = MagicMock()
    mock_res.json.return_value = {
        "query": {
            "pages": {
                "12345": {
                    "extract": "Full article content of Python programming language."
                }
            }
        }
    }
    mock_get.return_value = mock_res

    client = WikipediaClient(lang="en", user_agent="TestAgent/1.0")
    content = client.get_content("Python")

    assert content == "Full article content of Python programming language."


def test_wikipedia_client_ssrf_validation_error():
    with pytest.raises(ValueError, match="Invalid language code"):
        WikipediaClient(lang="en@hacker.site#")

    with pytest.raises(ValueError, match="Invalid language code"):
        WikipediaClient(lang="https://malicious.org")


@patch("requests.get")
def test_wikipedia_search_tool_uses_redirected_page_title_for_summary(
    mock_get, tool
):
    search_res = MagicMock()
    search_res.json.return_value = {"query": {"search": [{"title": "Py"}]}}

    page_redirect_res = MagicMock()
    page_redirect_res.json.return_value = {
        "query": {
            "redirects": [{"from": "Py", "to": "Python (programming language)"}],
            "pages": {"-1": {"missing": ""}},
        }
    }

    target_page_res = MagicMock()
    target_page_res.json.return_value = {
        "query": {
            "pages": {
                "12345": {
                    "pageid": 12345,
                    "title": "Python (programming language)",
                    "fullurl": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                }
            }
        }
    }

    summary_res = MagicMock()
    summary_res.json.return_value = {
        "query": {
            "pages": {"12345": {"extract": "Python is a high-level language."}}
        }
    }

    mock_get.side_effect = [
        search_res,
        page_redirect_res,
        target_page_res,
        summary_res,
    ]

    result = tool._run(search_query="Py")
    assert "Title: Python (programming language)" in result
    assert "Summary: Python is a high-level language." in result

    # Assert that summary request was called with the resolved title, not original query
    summary_call_params = mock_get.call_args_list[3].kwargs["params"]
    assert summary_call_params["titles"] == "Python (programming language)"


def test_wikipedia_search_tool_package_dependencies():
    tool = WikipediaSearchTool()
    assert tool.package_dependencies == ["beautifulsoup4", "requests"]


@patch("crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.logger")
@patch("crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WikipediaClient")
def test_wikipedia_search_security_logging(mock_client_cls, mock_logger, tool):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.search.side_effect = RuntimeError("Connection timeout")

    sensitive_query = "SECRET_PASSWORD_12345"
    result = tool._run(search_query=sensitive_query)

    assert sensitive_query in result  # User result contains query for user context
    mock_logger.error.assert_called_once_with(
        "Wikipedia search failed: %s", "RuntimeError"
    )


@patch("crewai_tools.tools.wikipedia_search_tool.wikipedia_search_tool.WIKIPEDIA_AVAILABLE", False)
def test_wikipedia_search_tool_missing_dependencies():
    with pytest.raises(ImportError) as exc_info:
        WikipediaSearchTool()
    assert "beautifulsoup4" in str(exc_info.value)
    assert "requests" in str(exc_info.value)

    tool = WikipediaSearchTool.__new__(WikipediaSearchTool)
    res = tool._run(search_query="Python")
    assert "beautifulsoup4" in res
    assert "requests" in res




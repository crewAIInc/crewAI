from pathlib import Path
from unittest.mock import MagicMock, patch
import urllib.error
import xml.etree.ElementTree as ET

from crewai_tools import ArxivPaperTool
import pytest


@pytest.fixture
def tool():
    return ArxivPaperTool(download_pdfs=False)


def mock_arxiv_response():
    return """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678</id>
                <title>Sample Paper</title>
                <summary>This is a summary of the sample paper.</summary>
                <published>2022-01-01T00:00:00Z</published>
                <author><name>John Doe</name></author>
                <link title="pdf" href="http://arxiv.org/pdf/1234.5678.pdf"/>
            </entry>
        </feed>"""


@patch("urllib.request.urlopen")
def test_fetch_arxiv_data(mock_urlopen, tool):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = mock_arxiv_response().encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    results = tool.fetch_arxiv_data("transformer", 1)
    assert isinstance(results, list)
    assert results[0]["title"] == "Sample Paper"


@patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Timeout"))
def test_fetch_arxiv_data_network_error(mock_urlopen, tool):
    with pytest.raises(urllib.error.URLError):
        tool.fetch_arxiv_data("transformer", 1)


@patch("urllib.request.urlretrieve")
def test_download_pdf_success(mock_urlretrieve):
    tool = ArxivPaperTool()
    tool.download_pdf("http://arxiv.org/pdf/1234.5678.pdf", Path("test.pdf"))
    mock_urlretrieve.assert_called_once()


@patch("urllib.request.urlretrieve", side_effect=OSError("Permission denied"))
def test_download_pdf_oserror(mock_urlretrieve):
    tool = ArxivPaperTool()
    with pytest.raises(OSError):
        tool.download_pdf(
            "http://arxiv.org/pdf/1234.5678.pdf", Path("/restricted/test.pdf")
        )


@patch("urllib.request.urlopen")
@patch("urllib.request.urlretrieve")
def test_run_with_download(mock_urlretrieve, mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = mock_arxiv_response().encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    tool = ArxivPaperTool(download_pdfs=True)
    output = tool._run("transformer", 1)
    assert "Title: Sample Paper" in output
    mock_urlretrieve.assert_called_once()


@patch("urllib.request.urlopen")
def test_run_no_download(mock_urlopen):
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = mock_arxiv_response().encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    tool = ArxivPaperTool(download_pdfs=False)
    result = tool._run("transformer", 1)
    assert "Title: Sample Paper" in result


@patch("pathlib.Path.mkdir")
def test_validate_save_path_creates_directory(mock_mkdir):
    path = ArxivPaperTool._validate_save_path("new_folder")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert isinstance(path, Path)


@patch("urllib.request.urlopen")
def test_run_handles_exception(mock_urlopen):
    mock_urlopen.side_effect = Exception("API failure")
    tool = ArxivPaperTool()
    result = tool._run("transformer", 1)
    assert "Failed to fetch or download Arxiv papers" in result


@patch("urllib.request.urlopen")
def test_invalid_xml_response(mock_urlopen, tool):
    mock_response = MagicMock()
    mock_response.read.return_value = b"<invalid><xml>"
    mock_response.status = 200
    mock_urlopen.return_value.__enter__.return_value = mock_response

    with pytest.raises(ET.ParseError):
        tool.fetch_arxiv_data("quantum", 1)


@patch.object(ArxivPaperTool, "fetch_arxiv_data")
def test_run_with_max_results(mock_fetch, tool):
    mock_fetch.return_value = [
        {
            "arxiv_id": f"test_{i}",
            "title": f"Title {i}",
            "summary": "Summary",
            "authors": ["Author"],
            "published_date": "2023-01-01",
            "pdf_url": None,
        }
        for i in range(100)
    ]

    result = tool._run(search_query="test", max_results=100)
    assert result.count("Title:") == 100

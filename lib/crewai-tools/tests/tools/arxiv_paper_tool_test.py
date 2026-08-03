from pathlib import Path
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

from crewai_tools import ArxivPaperTool
import pytest
import requests


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


@patch("crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_get")
def test_fetch_arxiv_data(mock_safe_get, tool):
    mock_response = MagicMock()
    mock_response.text = mock_arxiv_response()
    mock_safe_get.return_value = mock_response

    results = tool.fetch_arxiv_data("transformer", 1)
    assert isinstance(results, list)
    assert results[0]["title"] == "Sample Paper"


@patch(
    "crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_get",
    side_effect=requests.RequestException("Timeout"),
)
def test_fetch_arxiv_data_network_error(mock_safe_get, tool):
    with pytest.raises(requests.RequestException):
        tool.fetch_arxiv_data("transformer", 1)


@patch("crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_download")
def test_download_pdf_success(mock_safe_download):
    tool = ArxivPaperTool()
    tool.download_pdf("http://arxiv.org/pdf/1234.5678.pdf", Path("test.pdf"))
    mock_safe_download.assert_called_once_with(
        "http://arxiv.org/pdf/1234.5678.pdf",
        "test.pdf",
        timeout=ArxivPaperTool.REQUEST_TIMEOUT,
    )


@patch(
    "crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_download",
    side_effect=OSError("Permission denied"),
)
def test_download_pdf_oserror(mock_safe_download):
    tool = ArxivPaperTool()
    with pytest.raises(OSError):
        tool.download_pdf(
            "http://arxiv.org/pdf/1234.5678.pdf", Path("/restricted/test.pdf")
        )


def test_download_pdf_blocks_private_ip(monkeypatch):
    """Regression test for the SSRF fix (#6694): download_pdf must reject a
    URL that resolves to a private/reserved IP before making any request,
    end to end through the real safe_download/safe_get/validate_and_resolve
    chain -- not mocked out, so this exercises the actual protection.

    Clears CREWAI_TOOLS_ALLOW_UNSAFE_PATHS so this stays hermetic regardless
    of the environment: if that variable were set, validation would be
    skipped and this test would instead attempt a real request to a private
    address.
    """
    monkeypatch.delenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", raising=False)
    tool = ArxivPaperTool()
    with pytest.raises(ValueError, match="private/reserved IP"):
        tool.download_pdf("http://127.0.0.1/malicious.pdf", Path("test.pdf"))


def test_download_pdf_blocks_cloud_metadata_endpoint(monkeypatch):
    """Same as above, for the AWS/GCP/Azure metadata endpoint specifically --
    the concrete credential-theft scenario named in #6694."""
    monkeypatch.delenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", raising=False)
    tool = ArxivPaperTool()
    with pytest.raises(ValueError, match="private/reserved IP"):
        tool.download_pdf(
            "http://169.254.169.254/latest/meta-data/", Path("test.pdf")
        )


@patch("crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_get")
@patch("crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_download")
def test_run_with_download(mock_safe_download, mock_safe_get):
    mock_response = MagicMock()
    mock_response.text = mock_arxiv_response()
    mock_safe_get.return_value = mock_response

    tool = ArxivPaperTool(download_pdfs=True)
    output = tool._run("transformer", 1)
    assert "Title: Sample Paper" in output
    mock_safe_download.assert_called_once()


@patch("crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_get")
def test_run_no_download(mock_safe_get):
    mock_response = MagicMock()
    mock_response.text = mock_arxiv_response()
    mock_safe_get.return_value = mock_response

    tool = ArxivPaperTool(download_pdfs=False)
    result = tool._run("transformer", 1)
    assert "Title: Sample Paper" in result


@patch("pathlib.Path.mkdir")
def test_validate_save_path_creates_directory(mock_mkdir):
    path = ArxivPaperTool._validate_save_path("new_folder")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert isinstance(path, Path)


@patch(
    "crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_get",
    side_effect=Exception("API failure"),
)
def test_run_handles_exception(mock_safe_get):
    tool = ArxivPaperTool()
    result = tool._run("transformer", 1)
    assert "Failed to fetch or download Arxiv papers" in result


@patch("crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool.safe_get")
def test_invalid_xml_response(mock_safe_get, tool):
    mock_response = MagicMock()
    mock_response.text = "<invalid><xml>"
    mock_safe_get.return_value = mock_response

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

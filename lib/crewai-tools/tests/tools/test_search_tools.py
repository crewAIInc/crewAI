import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools import (
    CSVSearchTool,
    CodeDocsSearchTool,
    DOCXSearchTool,
    DirectorySearchTool,
    GithubSearchTool,
    JSONSearchTool,
    MDXSearchTool,
    PDFSearchTool,
    TXTSearchTool,
    WebsiteSearchTool,
    XMLSearchTool,
    YoutubeChannelSearchTool,
    YoutubeVideoSearchTool,
)
from crewai_tools.tools.rag.rag_tool import Adapter
import pytest


@pytest.fixture
def mock_adapter():
    mock_adapter = MagicMock(spec=Adapter)
    return mock_adapter


@pytest.mark.vcr()
def test_directory_search_tool():
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("This is a test file for directory search")

        tool = DirectorySearchTool(directory=temp_dir)
        result = tool._run(search_query="test file")
        assert "test file" in result.lower()


def test_pdf_search_tool(mock_adapter):
    mock_adapter.query.return_value = "this is a test"

    tool = PDFSearchTool(pdf="test.pdf", adapter=mock_adapter)
    result = tool._run(query="test content")
    assert "this is a test" in result.lower()
    mock_adapter.add.assert_called_once_with("test.pdf", data_type=DataType.PDF_FILE)
    mock_adapter.query.assert_called_once_with(
        "test content", similarity_threshold=0.6, limit=5
    )

    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = PDFSearchTool(adapter=mock_adapter)
    result = tool._run(pdf="test.pdf", query="test content")
    assert "this is a test" in result.lower()
    mock_adapter.add.assert_called_once_with("test.pdf", data_type=DataType.PDF_FILE)
    mock_adapter.query.assert_called_once_with(
        "test content", similarity_threshold=0.6, limit=5
    )


@pytest.mark.vcr()
def test_txt_search_tool():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_file.write(b"This is a test file for txt search")
        temp_file_path = temp_file.name

    try:
        tool = TXTSearchTool()
        tool.add(temp_file_path)
        result = tool._run(search_query="test file")
        assert "test file" in result.lower()
    finally:
        os.unlink(temp_file_path)


def test_docx_search_tool(mock_adapter):
    mock_adapter.query.return_value = "this is a test"

    tool = DOCXSearchTool(docx="test.docx", adapter=mock_adapter)
    result = tool._run(search_query="test content")
    assert "this is a test" in result.lower()
    mock_adapter.add.assert_called_once_with("test.docx", data_type=DataType.DOCX)
    mock_adapter.query.assert_called_once_with(
        "test content", similarity_threshold=0.6, limit=5
    )

    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = DOCXSearchTool(adapter=mock_adapter)
    result = tool._run(docx="test.docx", search_query="test content")
    assert "this is a test" in result.lower()
    mock_adapter.add.assert_called_once_with("test.docx", data_type=DataType.DOCX)
    mock_adapter.query.assert_called_once_with(
        "test content", similarity_threshold=0.6, limit=5
    )


@pytest.mark.vcr()
def test_json_search_tool():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(b'{"test": "This is a test JSON file"}')
        temp_file_path = temp_file.name

    try:
        tool = JSONSearchTool()
        result = tool._run(search_query="test JSON", json_path=temp_file_path)
        assert "test json" in result.lower()
    finally:
        os.unlink(temp_file_path)


def test_xml_search_tool(mock_adapter):
    mock_adapter.query.return_value = "this is a test"

    tool = XMLSearchTool(adapter=mock_adapter)
    result = tool._run(search_query="test XML", xml="test.xml")
    assert "this is a test" in result.lower()
    mock_adapter.add.assert_called_once_with("test.xml")
    mock_adapter.query.assert_called_once_with(
        "test XML", similarity_threshold=0.6, limit=5
    )


@pytest.mark.vcr()
def test_csv_search_tool():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file.write(b"name,description\ntest,This is a test CSV file")
        temp_file_path = temp_file.name

    try:
        tool = CSVSearchTool()
        tool.add(temp_file_path)
        result = tool._run(search_query="test CSV")
        assert "test csv" in result.lower()
    finally:
        os.unlink(temp_file_path)


@pytest.mark.vcr()
def test_mdx_search_tool():
    with tempfile.NamedTemporaryFile(suffix=".mdx", delete=False) as temp_file:
        temp_file.write(b"# Test MDX\nThis is a test MDX file")
        temp_file_path = temp_file.name

    try:
        tool = MDXSearchTool()
        tool.add(temp_file_path)
        result = tool._run(search_query="test MDX")
        assert "test mdx" in result.lower()
    finally:
        os.unlink(temp_file_path)


def test_website_search_tool(mock_adapter):
    mock_adapter.query.return_value = "this is a test"

    website = "https://crewai.com"
    search_query = "what is crewai?"
    tool = WebsiteSearchTool(website=website, adapter=mock_adapter)
    result = tool._run(search_query=search_query)

    mock_adapter.query.assert_called_once_with(
        "what is crewai?", similarity_threshold=0.6, limit=5
    )
    mock_adapter.add.assert_called_once_with(website, data_type=DataType.WEBSITE)

    assert "this is a test" in result.lower()

    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = WebsiteSearchTool(adapter=mock_adapter)
    result = tool._run(website=website, search_query=search_query)

    mock_adapter.query.assert_called_once_with(
        "what is crewai?", similarity_threshold=0.6, limit=5
    )
    mock_adapter.add.assert_called_once_with(website, data_type=DataType.WEBSITE)

    assert "this is a test" in result.lower()


def test_youtube_video_search_tool(mock_adapter):
    mock_adapter.query.return_value = "some video description"

    youtube_video_url = "https://www.youtube.com/watch?v=sample-video-id"
    search_query = "what is the video about?"
    tool = YoutubeVideoSearchTool(
        youtube_video_url=youtube_video_url,
        adapter=mock_adapter,
    )
    result = tool._run(search_query=search_query)
    assert "some video description" in result

    mock_adapter.add.assert_called_once_with(
        youtube_video_url, data_type=DataType.YOUTUBE_VIDEO
    )
    mock_adapter.query.assert_called_once_with(
        search_query, similarity_threshold=0.6, limit=5
    )

    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = YoutubeVideoSearchTool(adapter=mock_adapter)
    result = tool._run(youtube_video_url=youtube_video_url, search_query=search_query)
    assert "some video description" in result

    mock_adapter.add.assert_called_once_with(
        youtube_video_url, data_type=DataType.YOUTUBE_VIDEO
    )
    mock_adapter.query.assert_called_once_with(
        search_query, similarity_threshold=0.6, limit=5
    )


def test_youtube_channel_search_tool(mock_adapter):
    mock_adapter.query.return_value = "channel description"

    youtube_channel_handle = "@crewai"
    search_query = "what is the channel about?"
    tool = YoutubeChannelSearchTool(
        youtube_channel_handle=youtube_channel_handle, adapter=mock_adapter
    )
    result = tool._run(search_query=search_query)
    assert "channel description" in result
    mock_adapter.add.assert_called_once_with(
        youtube_channel_handle, data_type=DataType.YOUTUBE_CHANNEL
    )
    mock_adapter.query.assert_called_once_with(
        search_query, similarity_threshold=0.6, limit=5
    )

    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = YoutubeChannelSearchTool(adapter=mock_adapter)
    result = tool._run(
        youtube_channel_handle=youtube_channel_handle, search_query=search_query
    )
    assert "channel description" in result

    mock_adapter.add.assert_called_once_with(
        youtube_channel_handle, data_type=DataType.YOUTUBE_CHANNEL
    )
    mock_adapter.query.assert_called_once_with(
        search_query, similarity_threshold=0.6, limit=5
    )


def test_code_docs_search_tool(mock_adapter):
    mock_adapter.query.return_value = "test documentation"

    docs_url = "https://crewai.com/any-docs-url"
    search_query = "test documentation"
    tool = CodeDocsSearchTool(docs_url=docs_url, adapter=mock_adapter)
    result = tool._run(search_query=search_query)
    assert "test documentation" in result
    mock_adapter.add.assert_called_once_with(docs_url, data_type=DataType.DOCS_SITE)
    mock_adapter.query.assert_called_once_with(
        search_query, similarity_threshold=0.6, limit=5
    )

    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = CodeDocsSearchTool(adapter=mock_adapter)
    result = tool._run(docs_url=docs_url, search_query=search_query)
    assert "test documentation" in result
    mock_adapter.add.assert_called_once_with(docs_url, data_type=DataType.DOCS_SITE)
    mock_adapter.query.assert_called_once_with(
        search_query, similarity_threshold=0.6, limit=5
    )


def test_github_search_tool(mock_adapter):
    mock_adapter.query.return_value = "repo description"

    # ensure the provided repo and content types are used after initialization
    tool = GithubSearchTool(
        gh_token="test_token",
        github_repo="crewai/crewai",
        content_types=["code"],
        adapter=mock_adapter,
    )
    result = tool._run(search_query="tell me about crewai repo")
    assert "repo description" in result
    mock_adapter.add.assert_called_once_with(
        "https://github.com/crewai/crewai",
        data_type=DataType.GITHUB,
        metadata={"content_types": ["code"], "gh_token": "test_token"},
    )
    mock_adapter.query.assert_called_once_with(
        "tell me about crewai repo", similarity_threshold=0.6, limit=5
    )

    # ensure content types provided by run call is used
    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = GithubSearchTool(gh_token="test_token", adapter=mock_adapter)
    result = tool._run(
        github_repo="crewai/crewai",
        content_types=["code", "issue"],
        search_query="tell me about crewai repo",
    )
    assert "repo description" in result
    mock_adapter.add.assert_called_once_with(
        "https://github.com/crewai/crewai",
        data_type=DataType.GITHUB,
        metadata={"content_types": ["code", "issue"], "gh_token": "test_token"},
    )
    mock_adapter.query.assert_called_once_with(
        "tell me about crewai repo", similarity_threshold=0.6, limit=5
    )

    # ensure default content types are used if not provided
    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = GithubSearchTool(gh_token="test_token", adapter=mock_adapter)
    result = tool._run(
        github_repo="crewai/crewai",
        search_query="tell me about crewai repo",
    )
    assert "repo description" in result
    mock_adapter.add.assert_called_once_with(
        "https://github.com/crewai/crewai",
        data_type=DataType.GITHUB,
        metadata={
            "content_types": ["code", "repo", "pr", "issue"],
            "gh_token": "test_token",
        },
    )
    mock_adapter.query.assert_called_once_with(
        "tell me about crewai repo", similarity_threshold=0.6, limit=5
    )

    # ensure nothing is added if no repo is provided
    mock_adapter.query.reset_mock()
    mock_adapter.add.reset_mock()

    tool = GithubSearchTool(gh_token="test_token", adapter=mock_adapter)
    result = tool._run(search_query="tell me about crewai repo")
    mock_adapter.add.assert_not_called()
    mock_adapter.query.assert_called_once_with(
        "tell me about crewai repo", similarity_threshold=0.6, limit=5
    )

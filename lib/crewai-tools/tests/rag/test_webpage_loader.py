from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.webpage_loader import WebPageLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


class TestWebPageLoader:
    def setup_mock_response(self, text, status_code=200, content_type="text/html"):
        response = Mock()
        response.text = text
        response.apparent_encoding = "utf-8"
        response.status_code = status_code
        response.headers = {"content-type": content_type}
        return response

    def setup_mock_soup(self, text, title=None, script_style_elements=None):
        soup = Mock()
        soup.get_text.return_value = text
        soup.title = Mock(string=title) if title is not None else None
        soup.return_value = script_style_elements or []
        return soup

    @patch("requests.get")
    @patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup")
    def test_load_basic_webpage(self, mock_bs, mock_get):
        mock_get.return_value = self.setup_mock_response(
            "<html><head><title>Test Page</title></head><body><p>Test content</p></body></html>"
        )
        mock_bs.return_value = self.setup_mock_soup("Test content", title="Test Page")

        loader = WebPageLoader()
        result = loader.load(SourceContent("https://example.com"))

        assert isinstance(result, LoaderResult)
        assert result.content == "Test content"
        assert result.metadata["title"] == "Test Page"

    @patch("requests.get")
    @patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup")
    def test_load_webpage_with_scripts_and_styles(self, mock_bs, mock_get):
        html = """
        <html><head><title>Page with Scripts</title><style>body { color: red; }</style></head>
        <body><script>console.log('test');</script><p>Visible content</p></body></html>
        """
        mock_get.return_value = self.setup_mock_response(html)
        scripts = [Mock(), Mock()]
        styles = [Mock()]
        for el in scripts + styles:
            el.decompose = Mock()
        mock_bs.return_value = self.setup_mock_soup(
            "Page with Scripts Visible content",
            title="Page with Scripts",
            script_style_elements=scripts + styles,
        )

        loader = WebPageLoader()
        result = loader.load(SourceContent("https://example.com/with-scripts"))

        assert "Visible content" in result.content
        for el in scripts + styles:
            el.decompose.assert_called_once()

    @patch("requests.get")
    @patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup")
    def test_text_cleaning_and_title_handling(self, mock_bs, mock_get):
        mock_get.return_value = self.setup_mock_response(
            "<html><body><p>   Messy text </p></body></html>"
        )
        mock_bs.return_value = self.setup_mock_soup(
            "Text   with  extra spaces\n\n  More\t text  \n\n", title=None
        )

        loader = WebPageLoader()
        result = loader.load(SourceContent("https://example.com/messy-text"))
        assert result.content is not None
        assert result.metadata["title"] == ""

    @patch("requests.get")
    @patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup")
    def test_empty_or_missing_title(self, mock_bs, mock_get):
        for title in [None, ""]:
            mock_get.return_value = self.setup_mock_response(
                "<html><head><title></title></head><body>Content</body></html>"
            )
            mock_bs.return_value = self.setup_mock_soup("Content", title=title)

            loader = WebPageLoader()
            result = loader.load(SourceContent("https://example.com"))
            assert result.metadata["title"] == ""

    @patch("requests.get")
    def test_custom_and_default_headers(self, mock_get):
        mock_get.return_value = self.setup_mock_response(
            "<html><body>Test</body></html>"
        )
        custom_headers = {
            "User-Agent": "Bot",
            "Authorization": "Bearer xyz",
            "Accept": "text/html",
        }

        with patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup") as mock_bs:
            mock_bs.return_value = self.setup_mock_soup("Test")
            WebPageLoader().load(
                SourceContent("https://example.com"), headers=custom_headers
            )

        assert mock_get.call_args[1]["headers"] == custom_headers

    @patch("requests.get")
    def test_error_handling(self, mock_get):
        for error in [Exception("Fail"), ValueError("Bad"), ImportError("Oops")]:
            mock_get.side_effect = error
            with pytest.raises(ValueError, match="Error loading webpage"):
                WebPageLoader().load(SourceContent("https://example.com"))

    @patch("requests.get")
    def test_timeout_and_http_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("Timeout")
        with pytest.raises(ValueError):
            WebPageLoader().load(SourceContent("https://example.com"))

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.side_effect = None
        mock_get.return_value = mock_response
        with pytest.raises(ValueError):
            WebPageLoader().load(SourceContent("https://example.com/404"))

    @patch("requests.get")
    @patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup")
    def test_doc_id_consistency(self, mock_bs, mock_get):
        mock_get.return_value = self.setup_mock_response(
            "<html><body>Doc</body></html>"
        )
        mock_bs.return_value = self.setup_mock_soup("Doc")

        loader = WebPageLoader()
        result1 = loader.load(SourceContent("https://example.com"))
        result2 = loader.load(SourceContent("https://example.com"))

        assert result1.doc_id == result2.doc_id

    @patch("requests.get")
    @patch("crewai_tools.rag.loaders.webpage_loader.BeautifulSoup")
    def test_status_code_and_content_type(self, mock_bs, mock_get):
        for status in [200, 201, 301]:
            mock_get.return_value = self.setup_mock_response(
                f"<html><body>Status {status}</body></html>", status_code=status
            )
            mock_bs.return_value = self.setup_mock_soup(f"Status {status}")
            result = WebPageLoader().load(
                SourceContent(f"https://example.com/{status}")
            )
            assert result.metadata["status_code"] == status

        for ctype in ["text/html", "text/plain", "application/xhtml+xml"]:
            mock_get.return_value = self.setup_mock_response(
                "<html><body>Content</body></html>", content_type=ctype
            )
            mock_bs.return_value = self.setup_mock_soup("Content")
            result = WebPageLoader().load(SourceContent("https://example.com"))
            assert result.metadata["content_type"] == ctype

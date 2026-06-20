"""Tests for FileUrl source type and URL resolution."""

from unittest.mock import AsyncMock, MagicMock, patch

from crewai_files import FileBytes, FileUrl, ImageFile
from crewai_files.core.resolved import InlineBase64, UrlReference
from crewai_files.core.sources import FilePath, _normalize_source
from crewai_files.resolution.resolver import FileResolver
import pytest


def _addrinfo(ip: str) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    """Build a minimal ``socket.getaddrinfo`` result for a single IP.

    Args:
        ip: The IP address the host should resolve to.

    Returns:
        A one-entry list shaped like ``socket.getaddrinfo`` output.
    """
    return [(0, 0, 0, "", (ip, 0))]


@pytest.fixture(autouse=True)
def mock_public_dns():
    """Resolve every host to a public IP so fetch tests stay offline.

    The SSRF guard resolves the URL host; without this fixture the read tests
    would perform real DNS lookups. Individual tests can still override
    ``socket.getaddrinfo`` to exercise blocked addresses.

    Yields:
        The patched ``getaddrinfo`` mock.
    """
    with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")) as m:
        yield m


class TestFileUrl:
    """Tests for FileUrl source type."""

    def test_create_file_url(self):
        """Test creating FileUrl with valid URL."""
        url = FileUrl(url="https://example.com/image.png")

        assert url.url == "https://example.com/image.png"
        assert url.filename is None

    def test_create_file_url_with_filename(self):
        """Test creating FileUrl with custom filename."""
        url = FileUrl(url="https://example.com/image.png", filename="custom.png")

        assert url.url == "https://example.com/image.png"
        assert url.filename == "custom.png"

    def test_invalid_url_scheme_raises(self):
        """Test that non-http(s) URLs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            FileUrl(url="ftp://example.com/file.txt")

    def test_invalid_url_scheme_file_raises(self):
        """Test that file:// URLs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            FileUrl(url="file:///path/to/file.txt")

    def test_http_url_valid(self):
        """Test that HTTP URLs are valid."""
        url = FileUrl(url="http://example.com/image.jpg")

        assert url.url == "http://example.com/image.jpg"

    def test_https_url_valid(self):
        """Test that HTTPS URLs are valid."""
        url = FileUrl(url="https://example.com/image.jpg")

        assert url.url == "https://example.com/image.jpg"

    def test_content_type_guessing_png(self):
        """Test content type guessing for PNG files."""
        url = FileUrl(url="https://example.com/image.png")

        assert url.content_type == "image/png"

    def test_content_type_guessing_jpeg(self):
        """Test content type guessing for JPEG files."""
        url = FileUrl(url="https://example.com/photo.jpg")

        assert url.content_type == "image/jpeg"

    def test_content_type_guessing_pdf(self):
        """Test content type guessing for PDF files."""
        url = FileUrl(url="https://example.com/document.pdf")

        assert url.content_type == "application/pdf"

    def test_content_type_guessing_with_query_params(self):
        """Test content type guessing with URL query parameters."""
        url = FileUrl(url="https://example.com/image.png?v=123&token=abc")

        assert url.content_type == "image/png"

    def test_content_type_fallback_unknown(self):
        """Test content type falls back to octet-stream for unknown extensions."""
        url = FileUrl(url="https://example.com/file.unknownext123")

        assert url.content_type == "application/octet-stream"

    def test_content_type_no_extension(self):
        """Test content type for URL without extension."""
        url = FileUrl(url="https://example.com/file")

        assert url.content_type == "application/octet-stream"

    def test_read_fetches_content(self):
        """Test that read() fetches content from URL."""
        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"fake image content"
        mock_response.headers = {"content-type": "image/png"}
        mock_response.is_redirect = False

        with patch("httpx.get", return_value=mock_response) as mock_get:
            content = url.read()

            mock_get.assert_called_once_with(
                "https://example.com/image.png", follow_redirects=False
            )
            assert content == b"fake image content"

    def test_read_caches_content(self):
        """Test that read() caches content."""
        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"fake content"
        mock_response.headers = {}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            content1 = url.read()
            content2 = url.read()

            mock_get.assert_called_once()
            assert content1 == content2

    def test_read_updates_content_type_from_response(self):
        """Test that read() updates content type from response headers."""
        url = FileUrl(url="https://example.com/file")
        mock_response = MagicMock()
        mock_response.content = b"fake content"
        mock_response.headers = {"content-type": "image/webp; charset=utf-8"}

        with patch("httpx.get", return_value=mock_response):
            url.read()

            assert url.content_type == "image/webp"

    @pytest.mark.asyncio
    async def test_aread_fetches_content(self):
        """Test that aread() fetches content from URL asynchronously."""
        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"async fake content"
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            content = await url.aread()

            assert content == b"async fake content"

    @pytest.mark.asyncio
    async def test_aread_caches_content(self):
        """Test that aread() caches content."""
        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"cached content"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            content1 = await url.aread()
            content2 = await url.aread()

            mock_client.get.assert_called_once()
            assert content1 == content2


class TestNormalizeSource:
    """Tests for _normalize_source with URL detection."""

    def test_normalize_url_string(self):
        """Test that URL strings are converted to FileUrl."""
        result = _normalize_source("https://example.com/image.png")

        assert isinstance(result, FileUrl)
        assert result.url == "https://example.com/image.png"

    def test_normalize_http_url_string(self):
        """Test that HTTP URL strings are converted to FileUrl."""
        result = _normalize_source("http://example.com/file.pdf")

        assert isinstance(result, FileUrl)
        assert result.url == "http://example.com/file.pdf"

    def test_normalize_file_path_string(self, tmp_path):
        """Test that file path strings are converted to FilePath."""
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"test content")

        result = _normalize_source(str(test_file))

        assert isinstance(result, FilePath)

    def test_normalize_relative_path_is_not_url(self):
        """Test that relative path strings are not treated as URLs."""
        result = _normalize_source("https://example.com/file.png")

        assert isinstance(result, FileUrl)
        assert not isinstance(result, FilePath)

    def test_normalize_file_url_passthrough(self):
        """Test that FileUrl instances pass through unchanged."""
        original = FileUrl(url="https://example.com/image.png")
        result = _normalize_source(original)

        assert result is original


class TestResolverUrlHandling:
    """Tests for FileResolver URL handling."""

    def test_resolve_url_source_for_supported_provider(self):
        """Test URL source resolves to UrlReference for supported providers."""
        resolver = FileResolver()
        file = ImageFile(source=FileUrl(url="https://example.com/image.png"))

        resolved = resolver.resolve(file, "anthropic")

        assert isinstance(resolved, UrlReference)
        assert resolved.url == "https://example.com/image.png"
        assert resolved.content_type == "image/png"

    def test_resolve_url_source_openai(self):
        """Test URL source resolves to UrlReference for OpenAI."""
        resolver = FileResolver()
        file = ImageFile(source=FileUrl(url="https://example.com/photo.jpg"))

        resolved = resolver.resolve(file, "openai")

        assert isinstance(resolved, UrlReference)
        assert resolved.url == "https://example.com/photo.jpg"

    def test_resolve_url_source_gemini(self):
        """Test URL source resolves to UrlReference for Gemini."""
        resolver = FileResolver()
        file = ImageFile(source=FileUrl(url="https://example.com/image.webp"))

        resolved = resolver.resolve(file, "gemini")

        assert isinstance(resolved, UrlReference)
        assert resolved.url == "https://example.com/image.webp"

    def test_resolve_url_source_azure(self):
        """Test URL source resolves to UrlReference for Azure."""
        resolver = FileResolver()
        file = ImageFile(source=FileUrl(url="https://example.com/image.gif"))

        resolved = resolver.resolve(file, "azure")

        assert isinstance(resolved, UrlReference)
        assert resolved.url == "https://example.com/image.gif"

    def test_resolve_url_source_bedrock_fetches_content(self):
        """Test URL source fetches content for Bedrock (unsupported URLs)."""
        resolver = FileResolver()
        file_url = FileUrl(url="https://example.com/image.png")
        file = ImageFile(source=file_url)

        mock_response = MagicMock()
        mock_response.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        mock_response.headers = {"content-type": "image/png"}

        with patch("httpx.get", return_value=mock_response):
            resolved = resolver.resolve(file, "bedrock")

            assert not isinstance(resolved, UrlReference)

    def test_resolve_bytes_source_still_works(self):
        """Test that bytes source still resolves normally."""
        resolver = FileResolver()
        minimal_png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x08\x00\x00\x00\x08"
            b"\x01\x00\x00\x00\x00\xf9Y\xab\xcd\x00\x00\x00\nIDATx\x9cc`\x00\x00"
            b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        file = ImageFile(source=FileBytes(data=minimal_png, filename="test.png"))

        resolved = resolver.resolve(file, "anthropic")

        assert isinstance(resolved, InlineBase64)

    @pytest.mark.asyncio
    async def test_aresolve_url_source(self):
        """Test async URL resolution for supported provider."""
        resolver = FileResolver()
        file = ImageFile(source=FileUrl(url="https://example.com/image.png"))

        resolved = await resolver.aresolve(file, "anthropic")

        assert isinstance(resolved, UrlReference)
        assert resolved.url == "https://example.com/image.png"


class TestImageFileWithUrl:
    """Tests for creating ImageFile with URL source."""

    def test_image_file_from_url_string(self):
        """Test creating ImageFile from URL string."""
        file = ImageFile(source="https://example.com/image.png")

        assert isinstance(file.source, FileUrl)
        assert file.source.url == "https://example.com/image.png"

    def test_image_file_from_file_url(self):
        """Test creating ImageFile from FileUrl instance."""
        url = FileUrl(url="https://example.com/photo.jpg")
        file = ImageFile(source=url)

        assert file.source is url
        assert file.content_type == "image/jpeg"


class TestFileUrlSSRF:
    """SSRF protection for FileUrl.read / aread (CWE-918)."""

    @pytest.mark.parametrize(
        "blocked_ip",
        [
            "127.0.0.1",  # loopback
            "169.254.169.254",  # cloud metadata
            "10.0.0.5",  # RFC1918
            "192.168.1.10",  # RFC1918
            "::1",  # IPv6 loopback
            "::ffff:127.0.0.1",  # IPv4-mapped loopback (naive-guard bypass)
            "0.0.0.0",  # unspecified
        ],
    )
    def test_read_blocks_non_public_addresses(self, blocked_ip):
        """read() must refuse URLs resolving to a non-public address."""
        url = FileUrl(url="http://internal.example/secret")
        with patch("socket.getaddrinfo", return_value=_addrinfo(blocked_ip)):
            with patch("httpx.get") as mock_get:
                with pytest.raises(ValueError, match="SSRF protection"):
                    url.read()
                mock_get.assert_not_called()

    def test_read_allows_public_address(self):
        """read() must still fetch a normal public URL (no false positive)."""
        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"ok"
        mock_response.headers = {"content-type": "image/png"}
        mock_response.is_redirect = False

        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.get", return_value=mock_response) as mock_get:
                assert url.read() == b"ok"
                mock_get.assert_called_once_with(
                    "https://example.com/image.png", follow_redirects=False
                )

    def test_read_blocks_redirect_to_internal(self):
        """A public URL redirecting to an internal address must be blocked."""
        url = FileUrl(url="https://example.com/start")
        redirect = MagicMock()
        redirect.is_redirect = True
        redirect.headers = {"location": "http://169.254.169.254/latest/meta-data/"}

        def fake_getaddrinfo(host, *_args, **_kwargs):
            """Resolve the public start host and the internal redirect host.

            Args:
                host: The host being resolved.

            Returns:
                A ``getaddrinfo``-shaped result for the requested host.
            """
            mapping = {
                "example.com": "93.184.216.34",
                "169.254.169.254": "169.254.169.254",
            }
            return _addrinfo(mapping[host])

        with patch("socket.getaddrinfo", side_effect=fake_getaddrinfo):
            with patch("httpx.get", return_value=redirect):
                with pytest.raises(ValueError, match="SSRF protection"):
                    url.read()

    def test_read_blocks_redirect_bomb(self):
        """Endless redirects must raise rather than loop forever."""
        url = FileUrl(url="https://example.com/a")
        loop_response = MagicMock()
        loop_response.is_redirect = True
        loop_response.headers = {"location": "https://example.com/a"}

        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.get", return_value=loop_response):
                with pytest.raises(ValueError, match="Too many redirects"):
                    url.read()

    @pytest.mark.asyncio
    async def test_aread_blocks_non_public_address(self):
        """aread() must apply the same SSRF guard as read()."""
        url = FileUrl(url="http://internal.example/secret")
        mock_client = MagicMock()
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
            with patch("httpx.AsyncClient", return_value=mock_client):
                with pytest.raises(ValueError, match="SSRF protection"):
                    await url.aread()
                mock_client.get.assert_not_called()

"""Tests for FileUrl source type and URL resolution."""

from unittest.mock import AsyncMock, MagicMock, patch

from crewai_files import FileBytes, FileUrl, ImageFile
from crewai_files.core.resolved import InlineBase64, UrlReference
from crewai_files.core.sources import _MAX_REDIRECTS, FilePath, _normalize_source
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


def _response(
    content: bytes = b"",
    headers: dict[str, str] | None = None,
    *,
    is_redirect: bool = False,
) -> MagicMock:
    """Build a mock httpx response.

    Args:
        content: The response body.
        headers: The response headers.
        is_redirect: Whether the response is a redirect.

    Returns:
        A configured mock response.
    """
    response = MagicMock()
    response.content = content
    response.headers = headers if headers is not None else {}
    response.is_redirect = is_redirect
    response.raise_for_status = MagicMock()
    return response


def _sync_client(responses: list[MagicMock]) -> MagicMock:
    """Build a mock ``httpx.Client`` yielding ``responses`` in order.

    Args:
        responses: Responses returned by successive ``send`` calls.

    Returns:
        A mock client supporting the context-manager and request API.
    """
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.build_request = MagicMock(
        side_effect=lambda method, url, **kwargs: {
            "method": method,
            "url": url,
            **kwargs,
        }
    )
    client.send = MagicMock(side_effect=list(responses))
    return client


def _async_client(responses: list[MagicMock]) -> MagicMock:
    """Build a mock ``httpx.AsyncClient`` yielding ``responses`` in order.

    Args:
        responses: Responses returned by successive ``send`` calls.

    Returns:
        A mock async client supporting the async-context-manager and request API.
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.build_request = MagicMock(
        side_effect=lambda method, url, **kwargs: {
            "method": method,
            "url": url,
            **kwargs,
        }
    )
    client.send = AsyncMock(side_effect=list(responses))
    return client


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
        client = _sync_client(
            [_response(b"fake image content", {"content-type": "image/png"})]
        )

        with patch("httpx.Client", return_value=client):
            content = url.read()

            client.send.assert_called_once()
            assert content == b"fake image content"

    def test_read_caches_content(self):
        """Test that read() caches content."""
        url = FileUrl(url="https://example.com/image.png")
        client = _sync_client([_response(b"fake content")])

        with patch("httpx.Client", return_value=client):
            content1 = url.read()
            content2 = url.read()

            client.send.assert_called_once()
            assert content1 == content2

    def test_read_updates_content_type_from_response(self):
        """Test that read() updates content type from response headers."""
        url = FileUrl(url="https://example.com/file")
        client = _sync_client(
            [_response(b"fake content", {"content-type": "image/webp; charset=utf-8"})]
        )

        with patch("httpx.Client", return_value=client):
            url.read()

            assert url.content_type == "image/webp"

    @pytest.mark.asyncio
    async def test_aread_fetches_content(self):
        """Test that aread() fetches content from URL asynchronously."""
        url = FileUrl(url="https://example.com/image.png")
        client = _async_client(
            [_response(b"async fake content", {"content-type": "image/png"})]
        )

        with patch("httpx.AsyncClient", return_value=client):
            content = await url.aread()

            assert content == b"async fake content"

    @pytest.mark.asyncio
    async def test_aread_caches_content(self):
        """Test that aread() caches content."""
        url = FileUrl(url="https://example.com/image.png")
        client = _async_client([_response(b"cached content")])

        with patch("httpx.AsyncClient", return_value=client):
            content1 = await url.aread()
            content2 = await url.aread()

            client.send.assert_called_once()
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

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        client = _sync_client([_response(png_bytes, {"content-type": "image/png"})])

        with patch("httpx.Client", return_value=client):
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
    """SSRF protection for FileUrl.read / aread (CWE-918), incl. DNS rebinding."""

    @pytest.mark.parametrize(
        "blocked_ip",
        [
            "127.0.0.1",  # loopback
            "169.254.169.254",  # cloud metadata
            "10.0.0.5",  # RFC1918
            "192.168.1.10",  # RFC1918
            "172.16.0.1",  # RFC1918
            "::1",  # IPv6 loopback
            "::ffff:127.0.0.1",  # IPv4-mapped loopback (naive-guard bypass)
            "::ffff:169.254.169.254",  # IPv4-mapped metadata
            "fc00::1",  # IPv6 ULA (private)
            "fe80::1",  # IPv6 link-local
            "0.0.0.0",  # unspecified
            "224.0.0.1",  # multicast
        ],
    )
    def test_read_blocks_non_public_addresses(self, blocked_ip):
        """read() must refuse URLs resolving to a non-public address."""
        url = FileUrl(url="http://internal.example/secret")
        client = _sync_client([])
        with patch("socket.getaddrinfo", return_value=_addrinfo(blocked_ip)):
            with patch("httpx.Client", return_value=client):
                with pytest.raises(ValueError, match="SSRF protection"):
                    url.read()
                client.send.assert_not_called()

    def test_read_blocks_when_any_record_is_private(self):
        """A host with mixed public/private records must be rejected."""
        url = FileUrl(url="http://split.example/x")
        addrinfo = _addrinfo("93.184.216.34") + _addrinfo("127.0.0.1")
        client = _sync_client([])
        with patch("socket.getaddrinfo", return_value=addrinfo):
            with patch("httpx.Client", return_value=client):
                with pytest.raises(ValueError, match="SSRF protection"):
                    url.read()
                client.send.assert_not_called()

    def test_read_pins_connection_to_validated_ip(self):
        """read() connects to the validated IP, preserving Host and SNI."""
        url = FileUrl(url="https://example.com/image.png")
        client = _sync_client([_response(b"ok", {"content-type": "image/png"})])
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.Client", return_value=client):
                assert url.read() == b"ok"
        method, sent_url = client.build_request.call_args.args
        kwargs = client.build_request.call_args.kwargs
        assert method == "GET"
        assert sent_url == "https://93.184.216.34/image.png"
        assert kwargs["headers"]["Host"] == "example.com"
        assert kwargs["extensions"]["sni_hostname"] == "example.com"

    def test_read_does_not_re_resolve_hostname(self):
        """DNS rebinding cannot bypass the guard (host resolved once, IP pinned)."""
        url = FileUrl(url="http://rebind.example/x")
        getaddrinfo = MagicMock(return_value=_addrinfo("93.184.216.34"))
        client = _sync_client([_response(b"x")])
        with patch("socket.getaddrinfo", getaddrinfo):
            with patch("httpx.Client", return_value=client):
                url.read()
        assert getaddrinfo.call_count == 1
        _, sent_url = client.build_request.call_args.args
        assert "93.184.216.34" in sent_url
        assert "rebind.example" not in sent_url

    def test_read_preserves_explicit_port(self):
        """The validated-IP URL and Host header keep the explicit port."""
        url = FileUrl(url="https://example.com:8443/f")
        client = _sync_client([_response(b"ok")])
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.Client", return_value=client):
                url.read()
        _, sent_url = client.build_request.call_args.args
        kwargs = client.build_request.call_args.kwargs
        assert sent_url == "https://93.184.216.34:8443/f"
        assert kwargs["headers"]["Host"] == "example.com:8443"

    def test_read_brackets_ipv6_target(self):
        """An IPv6 connection target must be bracketed in the pinned URL."""
        url = FileUrl(url="https://v6.example/f")
        client = _sync_client([_response(b"ok")])
        with patch("socket.getaddrinfo", return_value=_addrinfo("2606:2800:220:1::1")):
            with patch("httpx.Client", return_value=client):
                url.read()
        _, sent_url = client.build_request.call_args.args
        assert sent_url == "https://[2606:2800:220:1::1]/f"

    def test_read_allows_public_address(self):
        """read() must still fetch a normal public URL (no false positive)."""
        url = FileUrl(url="https://example.com/image.png")
        client = _sync_client([_response(b"ok", {"content-type": "image/png"})])
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.Client", return_value=client):
                assert url.read() == b"ok"
                client.send.assert_called_once()

    def test_read_blocks_redirect_to_internal(self):
        """A public URL redirecting to an internal address must be blocked."""
        url = FileUrl(url="https://example.com/start")
        redirect = _response(
            headers={"location": "http://169.254.169.254/latest/meta-data/"},
            is_redirect=True,
        )
        client = _sync_client([redirect])

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
            with patch("httpx.Client", return_value=client):
                with pytest.raises(ValueError, match="SSRF protection"):
                    url.read()

    def test_read_blocks_redirect_bomb(self):
        """Endless redirects must raise rather than loop forever."""
        url = FileUrl(url="https://example.com/a")
        redirects = [
            _response(headers={"location": "https://example.com/a"}, is_redirect=True)
            for _ in range(_MAX_REDIRECTS + 2)
        ]
        client = _sync_client(redirects)
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.Client", return_value=client):
                with pytest.raises(ValueError, match="Too many redirects"):
                    url.read()

    @pytest.mark.asyncio
    async def test_aread_blocks_non_public_address(self):
        """aread() must apply the same SSRF guard as read()."""
        url = FileUrl(url="http://internal.example/secret")
        client = _async_client([])
        with patch("socket.getaddrinfo", return_value=_addrinfo("127.0.0.1")):
            with patch("httpx.AsyncClient", return_value=client):
                with pytest.raises(ValueError, match="SSRF protection"):
                    await url.aread()
                client.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_aread_pins_connection_to_validated_ip(self):
        """aread() connects to the validated IP, preserving Host and SNI."""
        url = FileUrl(url="https://example.com/image.png")
        client = _async_client([_response(b"ok", {"content-type": "image/png"})])
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.AsyncClient", return_value=client):
                assert await url.aread() == b"ok"
        _, sent_url = client.build_request.call_args.args
        kwargs = client.build_request.call_args.kwargs
        assert sent_url == "https://93.184.216.34/image.png"
        assert kwargs["headers"]["Host"] == "example.com"
        assert kwargs["extensions"]["sni_hostname"] == "example.com"

    @pytest.mark.asyncio
    async def test_aread_blocks_redirect_to_internal(self):
        """A public URL redirecting to an internal address must be blocked (async)."""
        url = FileUrl(url="https://example.com/start")
        redirect = _response(
            headers={"location": "http://169.254.169.254/latest/meta-data/"},
            is_redirect=True,
        )
        client = _async_client([redirect])

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
            with patch("httpx.AsyncClient", return_value=client):
                with pytest.raises(ValueError, match="SSRF protection"):
                    await url.aread()

    @pytest.mark.asyncio
    async def test_aread_blocks_redirect_bomb(self):
        """Endless redirects must raise rather than loop forever (async)."""
        url = FileUrl(url="https://example.com/a")
        redirects = [
            _response(headers={"location": "https://example.com/a"}, is_redirect=True)
            for _ in range(_MAX_REDIRECTS + 2)
        ]
        client = _async_client(redirects)
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.AsyncClient", return_value=client):
                with pytest.raises(ValueError, match="Too many redirects"):
                    await url.aread()

    def test_read_constructs_client_without_following_redirects(self):
        """read() must build the client with follow_redirects disabled.

        Redirects are followed and re-validated manually; letting httpx auto-follow
        would skip the per-hop SSRF check, so a regression to
        ``follow_redirects=True`` must fail this test.
        """
        url = FileUrl(url="https://example.com/image.png")
        client = _sync_client([_response(b"ok")])
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.Client", return_value=client) as client_cls:
                url.read()
        assert client_cls.call_args.kwargs.get("follow_redirects") is False

    @pytest.mark.asyncio
    async def test_aread_constructs_client_without_following_redirects(self):
        """aread() must build the async client with follow_redirects disabled."""
        url = FileUrl(url="https://example.com/image.png")
        client = _async_client([_response(b"ok")])
        with patch("socket.getaddrinfo", return_value=_addrinfo("93.184.216.34")):
            with patch("httpx.AsyncClient", return_value=client) as client_cls:
                await url.aread()
        assert client_cls.call_args.kwargs.get("follow_redirects") is False

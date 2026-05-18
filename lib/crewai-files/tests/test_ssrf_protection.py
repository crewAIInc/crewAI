"""Tests for SSRF protection in FileUrl and the security module."""

from unittest.mock import patch

from crewai_files import FileUrl
from crewai_files.core.security import (
    _is_private_or_reserved,
    validate_url,
)
import pytest


class TestIsPrivateOrReserved:
    """Tests for _is_private_or_reserved helper."""

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "127.0.0.2",
            "10.0.0.1",
            "10.255.255.255",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.0.1",
            "192.168.1.100",
            "169.254.169.254",  # AWS metadata
            "0.0.0.0",
        ],
    )
    def test_private_ipv4_addresses_blocked(self, ip):
        assert _is_private_or_reserved(ip) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "::1",
            "::",
            "fc00::1",
            "fd00::1",
            "fe80::1",
        ],
    )
    def test_private_ipv6_addresses_blocked(self, ip):
        assert _is_private_or_reserved(ip) is True

    def test_ipv4_mapped_ipv6_loopback_blocked(self):
        assert _is_private_or_reserved("::ffff:127.0.0.1") is True

    def test_ipv4_mapped_ipv6_metadata_blocked(self):
        assert _is_private_or_reserved("::ffff:169.254.169.254") is True

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "93.184.216.34",  # example.com
            "2606:2800:220:1:248:1893:25c8:1946",
        ],
    )
    def test_public_addresses_allowed(self, ip):
        assert _is_private_or_reserved(ip) is False

    def test_unparseable_ip_blocked(self):
        assert _is_private_or_reserved("not-an-ip") is True


class TestValidateUrl:
    """Tests for validate_url function."""

    def test_blocks_file_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("file:///etc/passwd")

    def test_blocks_ftp_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("ftp://example.com/file")

    def test_blocks_data_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            validate_url("data:text/plain;base64,SGVsbG8=")

    def test_blocks_no_hostname(self):
        with pytest.raises(ValueError, match="URL has no hostname"):
            validate_url("http://")

    def test_blocks_localhost(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://localhost/secret")

    def test_blocks_127_0_0_1(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://127.0.0.1/secret")

    def test_blocks_metadata_endpoint(self):
        """Block AWS/GCP/Azure cloud metadata endpoint."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_private_10_network(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://10.0.0.1/internal")

    def test_blocks_private_172_network(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://172.16.0.1/internal")

    def test_blocks_private_192_168_network(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://192.168.1.1/internal")

    def test_blocks_0_0_0_0(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            validate_url("http://0.0.0.0/")

    def test_allows_public_https_url(self):
        result = validate_url("https://example.com/image.png")
        assert result == "https://example.com/image.png"

    def test_allows_public_http_url(self):
        result = validate_url("http://example.com/file.pdf")
        assert result == "http://example.com/file.pdf"

    def test_blocks_unresolvable_hostname(self):
        with pytest.raises(ValueError, match="Could not resolve hostname"):
            validate_url("https://this-domain-definitely-does-not-exist-12345.invalid/")

    def test_blocks_dns_resolving_to_private_ip(self):
        """Simulate a hostname that resolves to a private IP (DNS rebinding)."""
        fake_addrinfo = [(2, 1, 6, "", ("10.0.0.1", 443))]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            with pytest.raises(ValueError, match="private/reserved IP"):
                validate_url("https://evil.example.com/steal")

    def test_escape_hatch_allows_private_urls(self):
        """CREWAI_FILES_ALLOW_UNSAFE_URLS=true bypasses validation."""
        with patch.dict("os.environ", {"CREWAI_FILES_ALLOW_UNSAFE_URLS": "true"}):
            result = validate_url("http://127.0.0.1/secret")
            assert result == "http://127.0.0.1/secret"


class TestFileUrlSSRF:
    """Tests for SSRF protection integrated into FileUrl."""

    def test_rejects_localhost_url(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            FileUrl(url="http://localhost/secret")

    def test_rejects_127_0_0_1(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            FileUrl(url="http://127.0.0.1:8080/admin")

    def test_rejects_metadata_endpoint(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            FileUrl(url="http://169.254.169.254/latest/meta-data/")

    def test_rejects_private_network_10(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            FileUrl(url="http://10.0.0.1/internal-service")

    def test_rejects_private_network_172(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            FileUrl(url="http://172.16.5.10/api")

    def test_rejects_private_network_192_168(self):
        with pytest.raises(ValueError, match="private/reserved IP"):
            FileUrl(url="http://192.168.1.1/router")

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            FileUrl(url="file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="Invalid URL scheme"):
            FileUrl(url="ftp://example.com/file")

    def test_accepts_public_https_url(self):
        url = FileUrl(url="https://example.com/image.png")
        assert url.url == "https://example.com/image.png"

    def test_accepts_public_http_url(self):
        url = FileUrl(url="http://example.com/file.pdf")
        assert url.url == "http://example.com/file.pdf"

    def test_rejects_dns_rebinding_to_private_ip(self):
        """A hostname that resolves to an internal IP should be blocked."""
        fake_addrinfo = [(2, 1, 6, "", ("169.254.169.254", 80))]
        with patch("socket.getaddrinfo", return_value=fake_addrinfo):
            with pytest.raises(ValueError, match="private/reserved IP"):
                FileUrl(url="http://metadata.evil.com/steal-creds")

    def test_escape_hatch_bypasses_ssrf_check(self):
        with patch.dict("os.environ", {"CREWAI_FILES_ALLOW_UNSAFE_URLS": "true"}):
            url = FileUrl(url="http://127.0.0.1/local")
            assert url.url == "http://127.0.0.1/local"

    def test_read_does_not_follow_redirects(self):
        """Verify read() uses follow_redirects=False to prevent redirect-based SSRF."""
        from unittest.mock import MagicMock

        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_response.headers = {}

        with patch("httpx.get", return_value=mock_response) as mock_get:
            url.read()
            mock_get.assert_called_once_with(
                "https://example.com/image.png", follow_redirects=False
            )

    @pytest.mark.asyncio
    async def test_aread_does_not_follow_redirects(self):
        """Verify aread() uses follow_redirects=False."""
        from unittest.mock import AsyncMock, MagicMock

        url = FileUrl(url="https://example.com/image.png")
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await url.aread()
            mock_client.get.assert_called_once_with(
                "https://example.com/image.png", follow_redirects=False
            )

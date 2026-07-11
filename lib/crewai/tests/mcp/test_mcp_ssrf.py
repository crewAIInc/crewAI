"""Tests for SSRF protection in MCP tool wrappers."""

from __future__ import annotations

import socket
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.tools.mcp_tool_wrapper import (
    _validate_mcp_server_url,
    _validate_mcp_tool_args_for_urls,
)


@pytest.fixture
def mock_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock DNS to resolve example.com to a public IP."""
    original_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(
        host: str, port: int, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        if host in {"public.example", "example.com"}:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", port),
                )
            ]
        return original_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


class TestValidateMCPServerURL:
    """Tests for MCP server URL validation."""

    def test_blocks_internal_server_url(self) -> None:
        """MCP server URL pointing to localhost should be blocked."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_server_url("http://127.0.0.1:8080/mcp")

    def test_blocks_metadata_server_url(self) -> None:
        """MCP server URL pointing to cloud metadata should be blocked."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_server_url("http://169.254.169.254/latest/meta-data/")

    def test_blocks_file_scheme(self) -> None:
        """file:// MCP server URL should be blocked."""
        with pytest.raises(ValueError, match="file://"):
            _validate_mcp_server_url("file:///etc/passwd")

    def test_allows_public_server_url(self, mock_dns: None) -> None:
        """MCP server URL on a public IP should be allowed."""
        _validate_mcp_server_url("http://public.example:8080/mcp")


class TestValidateMCPToolArgsForURLs:
    """Tests for URL validation in MCP tool arguments."""

    def test_blocks_internal_url_in_string_arg(self) -> None:
        """URL in a string argument pointing to internal IP should be blocked."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_tool_args_for_urls({"url": "http://127.0.0.1/admin"})

    def test_blocks_metadata_url_in_string_arg(self) -> None:
        """URL in a string argument pointing to cloud metadata should be blocked."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_tool_args_for_urls(
                {"url": "http://169.254.169.254/latest/meta-data/"}
            )

    def test_blocks_url_in_nested_dict(self, mock_dns: None) -> None:
        """URL nested inside a dict argument should be validated."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_tool_args_for_urls(
                {"config": {"target": "http://10.0.0.1/internal"}}
            )

    def test_blocks_url_in_list(self, mock_dns: None) -> None:
        """URL inside a list argument should be validated."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_tool_args_for_urls(
                {"urls": ["http://192.168.1.1/router", "http://example.com"]}
            )

    def test_allows_public_url(self, mock_dns: None) -> None:
        """URL pointing to a public IP should be allowed."""
        _validate_mcp_tool_args_for_urls({"url": "http://public.example/data"})

    def test_allows_non_url_strings(self) -> None:
        """Non-URL strings should pass through without error."""
        _validate_mcp_tool_args_for_urls({"query": "search for python docs"})

    def test_allows_empty_args(self) -> None:
        """Empty arguments should pass through without error."""
        _validate_mcp_tool_args_for_urls({})

    def test_blocks_file_scheme_in_args(self) -> None:
        """file:// URLs in arguments should be blocked."""
        with pytest.raises(ValueError, match="file://"):
            _validate_mcp_tool_args_for_urls({"path": "file:///etc/passwd"})

    def test_validates_multiple_urls(self, mock_dns: None) -> None:
        """Multiple URLs in the same argument should all be validated."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_tool_args_for_urls(
                {
                    "urls": [
                        "http://public.example/page",
                        "http://172.16.0.1/internal",
                    ]
                }
            )

    def test_validates_url_in_list_of_dicts(self, mock_dns: None) -> None:
        """URLs inside a list of dicts should be validated."""
        with pytest.raises(ValueError, match="private/reserved IP"):
            _validate_mcp_tool_args_for_urls(
                {"items": [{"url": "http://192.168.0.1/admin"}]}
            )

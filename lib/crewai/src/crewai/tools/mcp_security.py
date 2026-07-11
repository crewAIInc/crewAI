"""Shared SSRF validation helpers for MCP tool wrappers.

Both ``mcp_tool_wrapper`` and ``mcp_native_tool`` need the same URL
validation logic. This module extracts it into a single location to
avoid divergence.
"""

import re
from typing import Any


_URL_PATTERN = re.compile(r"[a-zA-Z]+://[^\s\"']+")


def validate_mcp_server_url(url: str) -> None:
    """Validate that the MCP server URL is not an internal/reserved address.

    Raises:
        ValueError: If the URL resolves to a private or reserved IP.
    """
    from crewai_tools.security.safe_path import validate_url

    validate_url(url)


def validate_mcp_tool_args_for_urls(kwargs: dict[str, Any]) -> None:
    """Scan MCP tool arguments for URLs and validate them against SSRF rules.

    Recursively scans string values for http/https URLs and validates each
    one. This prevents agents from using MCP tools to access internal
    services, cloud metadata endpoints, or other private resources.

    Raises:
        ValueError: If any URL in the arguments resolves to a private/reserved IP.
    """
    from crewai_tools.security.safe_path import validate_url_and_resolve

    for value in kwargs.values():
        if isinstance(value, str):
            for match in _URL_PATTERN.finditer(value):
                validate_url_and_resolve(match.group())
        elif isinstance(value, dict):
            validate_mcp_tool_args_for_urls(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    for match in _URL_PATTERN.finditer(item):
                        validate_url_and_resolve(match.group())
                elif isinstance(item, dict):
                    validate_mcp_tool_args_for_urls(item)

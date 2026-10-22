"""Utility functions for RAG loaders."""

from typing import Any

from requests import Response


def _declares_charset(content_type: str) -> bool:
    """Return whether a Content-Type header value has a charset parameter."""
    return any(
        param.split("=", 1)[0].strip().lower() == "charset"
        for param in content_type.split(";")[1:]
    )


def _prefer_utf8(response: Response) -> None:
    """Read a ``text/*`` body as UTF-8 when the server declares no charset.

    Requests falls back to ISO-8859-1 for ``text/*`` responses without a
    charset, which garbles UTF-8 bodies. A declared charset is left as it is,
    and so is the fallback when the body is not valid UTF-8.
    """
    if response.encoding != "ISO-8859-1" or _declares_charset(
        response.headers.get("Content-Type", "")
    ):
        return
    try:
        response.content.decode("utf-8")
    except UnicodeDecodeError:
        return
    response.encoding = "utf-8-sig"


def load_from_url(
    url: str,
    kwargs: dict[str, Any],
    accept_header: str = "*/*",
    loader_name: str = "Loader",
) -> str:
    """Load content from a URL.

    Args:
        url: The URL to fetch content from
        kwargs: Additional keyword arguments (can include 'headers' override)
        accept_header: The Accept header value for the request
        loader_name: The name of the loader for the User-Agent header

    Returns:
        The text content from the URL

    Raises:
        ValueError: If there's an error fetching the URL
    """
    from crewai_tools.security.safe_requests import safe_get

    headers = kwargs.get(
        "headers",
        {
            "Accept": accept_header,
            "User-Agent": f"Mozilla/5.0 (compatible; crewai-tools {loader_name})",
        },
    )

    try:
        response = safe_get(url, headers=headers, timeout=30)
        _prefer_utf8(response)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise ValueError(f"Error fetching content from URL {url}: {e!s}") from e

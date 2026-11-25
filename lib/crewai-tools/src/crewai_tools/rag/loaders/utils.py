"""Utility functions for RAG loaders."""

from typing import Any

from requests import Response


def _declares_charset(content_type: str) -> bool:
    """Return whether a Content-Type header value has a charset parameter."""
    return any(
        param.split("=", 1)[0].strip().lower() == "charset"
        for param in content_type.split(";")[1:]
    )


def _response_text(response: Response) -> str:
    """Return the body of ``response`` as text.

    Requests falls back to ISO-8859-1 for ``text/*`` responses without a
    charset, which garbles UTF-8 bodies. In that case a body that is valid
    UTF-8 is decoded as UTF-8 (dropping a leading BOM). A declared charset is
    used as given, and a body that is not valid UTF-8 keeps the fallback.
    """
    if response.encoding == "ISO-8859-1" and not _declares_charset(
        response.headers.get("Content-Type", "")
    ):
        try:
            return response.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass
    return response.text


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
        response.raise_for_status()
        return _response_text(response)
    except Exception as e:
        raise ValueError(f"Error fetching content from URL {url}: {e!s}") from e

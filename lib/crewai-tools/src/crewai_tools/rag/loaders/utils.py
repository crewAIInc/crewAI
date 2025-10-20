"""Utility functions for RAG loaders."""


def load_from_url(
    url: str, kwargs: dict, accept_header: str = "*/*", loader_name: str = "Loader"
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
    import requests

    headers = kwargs.get(
        "headers",
        {
            "Accept": accept_header,
            "User-Agent": f"Mozilla/5.0 (compatible; crewai-tools {loader_name})",
        },
    )

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise ValueError(f"Error fetching content from URL {url}: {e!s}") from e

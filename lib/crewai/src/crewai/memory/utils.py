"""Utility functions for the unified memory system."""

from __future__ import annotations

import re
import unicodedata


def sanitize_scope_name(name: str) -> str:
    """Sanitize a name for use in hierarchical scope paths.

    Converts to lowercase, replaces every character that is not a Unicode
    letter, number, combining mark, underscore or hyphen with a hyphen,
    collapses multiple hyphens, strips leading/trailing hyphens. Non-ASCII
    letters (e.g. CJK, Hangul, accented Latin) and the combining marks of
    scripts such as Devanagari and Thai are kept so distinct names stay distinct.

    Args:
        name: The raw name to sanitize (e.g. crew name, agent role, flow class name).

    Returns:
        A sanitized string safe for use in scope paths. Returns 'unknown' if the
        result would be empty.

    Examples:
        >>> sanitize_scope_name("Research Crew")
        'research-crew'
        >>> sanitize_scope_name("Agent #1 (Main)")
        'agent-1-main'
        >>> sanitize_scope_name("café_worker")
        'café_worker'
        >>> sanitize_scope_name("東京支社")
        '東京支社'
    """
    if not name:
        return "unknown"
    # NFC so visually identical names (e.g. macOS NFD file names) map to one scope.
    name = unicodedata.normalize("NFC", name.lower().strip())
    # Keep letters (L), numbers (N) and combining marks (M). Marks matter:
    # Python's \w drops them, which would merge e.g. "काली" and "किली".
    name = "".join(
        c if c in "_-" or unicodedata.category(c)[0] in "LNM" else "-" for c in name
    )
    name = re.sub(r"-+", "-", name)
    name = name.strip("-")
    return name or "unknown"


def normalize_scope_path(path: str) -> str:
    """Normalize a scope path by removing double slashes and ensuring proper format.

    Args:
        path: The raw scope path (e.g. '/crew/MyCrewName//agent//role').

    Returns:
        A normalized path with leading slash, no trailing slash, no double slashes.
        Returns '/' for empty or root-only paths.

    Examples:
        >>> normalize_scope_path("/crew/test//agent//")
        '/crew/test/agent'
        >>> normalize_scope_path("")
        '/'
        >>> normalize_scope_path("crew/test")
        '/crew/test'
    """
    if not path or path == "/":
        return "/"
    path = re.sub(r"/+", "/", path)
    if not path.startswith("/"):
        path = "/" + path
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def join_scope_paths(root: str | None, inner: str | None) -> str:
    """Join a root scope with an inner scope, handling edge cases properly.

    Args:
        root: The root scope prefix (e.g. '/crew/research-crew').
        inner: The inner scope (e.g. '/market-trends' or 'market-trends').

    Returns:
        The combined, normalized scope path.

    Examples:
        >>> join_scope_paths("/crew/test", "/market-trends")
        '/crew/test/market-trends'
        >>> join_scope_paths("/crew/test", "market-trends")
        '/crew/test/market-trends'
        >>> join_scope_paths("/crew/test", "/")
        '/crew/test'
        >>> join_scope_paths("/crew/test", None)
        '/crew/test'
        >>> join_scope_paths(None, "/market-trends")
        '/market-trends'
        >>> join_scope_paths(None, None)
        '/'
    """
    root = root.rstrip("/") if root else ""
    inner = inner.strip("/") if inner else ""

    if root and inner:
        result = f"{root}/{inner}"
    elif root:
        result = root
    elif inner:
        result = f"/{inner}"
    else:
        result = "/"

    return normalize_scope_path(result)

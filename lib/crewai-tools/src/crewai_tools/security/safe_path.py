"""Path and URL validation utilities for crewai-tools.

Provides validation for file paths and URLs to prevent unauthorized
file access and server-side request forgery (SSRF) when tools accept
user-controlled or LLM-controlled inputs at runtime.

Set CREWAI_TOOLS_ALLOW_UNSAFE_PATHS=true to bypass validation (not
recommended for production).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

_UNSAFE_PATHS_ENV = "CREWAI_TOOLS_ALLOW_UNSAFE_PATHS"
_ALLOWED_DIRS_ENV = "CREWAI_TOOLS_ALLOWED_DIRS"


def _get_allowed_roots(
    base_dir: str | None = None,
    allowed_dirs: list[str] | None = None,
) -> list[str]:
    """Build the deny-by-default set of allowed root directories.

    Roots are drawn from, in order:

    1. ``base_dir`` (defaults to the current working directory),
    2. the ``CREWAI_TOOLS_ALLOWED_DIRS`` environment variable, split on
       ``os.pathsep``,
    3. the caller-supplied ``allowed_dirs`` list.

    Every root is resolved with :func:`os.path.realpath` so a symlinked root
    is compared by its real location. Empty entries are ignored and duplicates
    are collapsed while preserving order. The first element is always the
    primary root used to resolve relative candidate paths.

    The filesystem root (``os.sep``, e.g. ``"/"``) is never accepted as an
    *implicitly defaulted* root. When ``base_dir`` is not supplied and the
    current working directory is ``/`` -- common in containers started without
    a ``WORKDIR`` -- defaulting to it would make every absolute path "within"
    the allow-list and disable confinement entirely. In that case the cwd
    default is dropped; an operator who genuinely wants the whole filesystem
    must opt in explicitly via ``base_dir``, ``allowed_dirs``, or
    ``CREWAI_TOOLS_ALLOWED_DIRS``. If no usable root remains, a ``ValueError``
    is raised rather than silently allowing everything.
    """
    primary_explicit = base_dir is not None
    primary = base_dir if base_dir is not None else os.getcwd()

    # (root, is_explicit) -- explicit roots are operator-provided and may
    # legitimately include the filesystem root as an opt-in.
    raw_roots: list[tuple[str, bool]] = [(primary, primary_explicit)]

    env_dirs = os.environ.get(_ALLOWED_DIRS_ENV, "")
    if env_dirs:
        raw_roots.extend((d, True) for d in env_dirs.split(os.pathsep) if d)

    if allowed_dirs:
        raw_roots.extend((d, True) for d in allowed_dirs if d)

    resolved: list[str] = []
    seen: set[str] = set()
    for root, is_explicit in raw_roots:
        real = os.path.realpath(root)
        if real == os.sep and not is_explicit:
            # Refuse to let an unconfigured cwd of "/" open the whole filesystem.
            continue
        if real not in seen:
            seen.add(real)
            resolved.append(real)

    if not resolved:
        raise ValueError(
            "No safe allowed directory could be determined: the current working "
            f"directory is the filesystem root ('{os.sep}'). Set "
            f"{_ALLOWED_DIRS_ENV} to an explicit directory, pass "
            f"base_dir/allowed_dirs, or set {_UNSAFE_PATHS_ENV}=true to bypass "
            "path validation."
        )
    return resolved


def _is_within_root(resolved_path: str, resolved_root: str) -> bool:
    """Return True if *resolved_path* equals *resolved_root* or lives beneath it.

    When ``resolved_root`` already ends with a separator (e.g. the filesystem
    root ``"/"``), appending ``os.sep`` would double it, so the root is used
    as-is for the prefix in that case.
    """
    prefix = resolved_root if resolved_root.endswith(os.sep) else resolved_root + os.sep
    return resolved_path == resolved_root or resolved_path.startswith(prefix)


def format_path_for_display(path: str, base_dir: str | None = None) -> str:
    """Return a path label that does not expose absolute directory prefixes."""
    if base_dir is None:
        base_dir = os.getcwd()

    try:
        resolved_base = os.path.realpath(base_dir)
        resolved_path = os.path.realpath(
            os.path.join(resolved_base, path) if not os.path.isabs(path) else path
        )
        if os.path.commonpath([resolved_base, resolved_path]) == resolved_base:
            return os.path.relpath(resolved_path, resolved_base)
    except (OSError, ValueError) as exc:
        logger.debug("Falling back to basename for display path formatting: %s", exc)

    return os.path.basename(os.path.realpath(path)) or "[redacted path]"


def format_error_for_display(error: Exception) -> str:
    """Return exception details without OS-added absolute path context."""
    if isinstance(error, OSError):
        return error.strerror or error.__class__.__name__
    return str(error)


def _is_escape_hatch_enabled() -> bool:
    """Check if the unsafe paths escape hatch is enabled."""
    return os.environ.get(_UNSAFE_PATHS_ENV, "").lower() in ("true", "1", "yes")


def validate_file_path(
    path: str,
    base_dir: str | None = None,
    *,
    allowed_dirs: list[str] | None = None,
) -> str:
    """Validate that a file path is safe to read.

    Resolves symlinks and ``..`` components, then checks that the resolved
    path falls within at least one allowed root directory. The allow-list is
    built from *base_dir* (defaults to the current working directory), the
    ``CREWAI_TOOLS_ALLOWED_DIRS`` environment variable, and *allowed_dirs* —
    see :func:`_get_allowed_roots`. Access is denied by default for anything
    outside that set.

    Args:
        path: The file path to validate.
        base_dir: Primary allowed root. Defaults to ``os.getcwd()`` and is
            used to resolve relative ``path`` values.
        allowed_dirs: Additional allowed root directories.

    Returns:
        The resolved, validated absolute path.

    Raises:
        ValueError: If the path escapes every allowed directory.
    """
    if _is_escape_hatch_enabled():
        logger.warning(
            "%s is enabled — skipping file path validation for: %s",
            _UNSAFE_PATHS_ENV,
            path,
        )
        return os.path.realpath(path)

    allowed_roots = _get_allowed_roots(base_dir, allowed_dirs)
    primary_root = allowed_roots[0]

    resolved_path = os.path.realpath(
        path if os.path.isabs(path) else os.path.join(primary_root, path)
    )

    if any(_is_within_root(resolved_path, root) for root in allowed_roots):
        return resolved_path

    raise ValueError(
        f"Path '{format_path_for_display(resolved_path, primary_root)}' is "
        f"outside the allowed directories. "
        f"Add the directory via {_ALLOWED_DIRS_ENV}, or set "
        f"{_UNSAFE_PATHS_ENV}=true to bypass this check."
    )


def validate_directory_path(
    path: str,
    base_dir: str | None = None,
    *,
    allowed_dirs: list[str] | None = None,
) -> str:
    """Validate that a directory path is safe to read.

    Same as :func:`validate_file_path` but also checks that the path
    is an existing directory.

    Args:
        path: The directory path to validate.
        base_dir: Primary allowed root. Defaults to ``os.getcwd()``.
        allowed_dirs: Additional allowed root directories.

    Returns:
        The resolved, validated absolute path.

    Raises:
        ValueError: If the path escapes every allowed directory or is not a directory.
    """
    validated = validate_file_path(path, base_dir, allowed_dirs=allowed_dirs)
    if not os.path.isdir(validated):
        raise ValueError(f"Path '{validated}' is not a directory.")
    return validated


# Private and reserved IP ranges that should not be accessed
_BLOCKED_IPV4_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata
    ipaddress.ip_network("0.0.0.0/32"),
]

_BLOCKED_IPV6_NETWORKS = [
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("fc00::/7"),  # Unique local addresses
    ipaddress.ip_network("fe80::/10"),  # Link-local IPv6
]


def _is_private_or_reserved(ip_str: str) -> bool:
    """Check if an IP address is private, reserved, or otherwise unsafe."""
    try:
        addr = ipaddress.ip_address(ip_str)
        # Unwrap IPv4-mapped IPv6 addresses (e.g., ::ffff:127.0.0.1) to IPv4
        # so they are only checked against IPv4 networks (avoids TypeError when
        # an IPv4Address is compared against an IPv6Network).
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        networks = (
            _BLOCKED_IPV4_NETWORKS
            if isinstance(addr, ipaddress.IPv4Address)
            else _BLOCKED_IPV6_NETWORKS
        )
        return any(addr in network for network in networks)
    except ValueError:
        return True  # If we can't parse, block it


def validate_url(url: str) -> str:
    """Validate that a URL is safe to fetch.

    Blocks ``file://`` scheme entirely. For ``http``/``https``, resolves
    DNS and checks that the target IP is not private or reserved (prevents
    SSRF to internal services and cloud metadata endpoints).

    Args:
        url: The URL to validate.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL uses a blocked scheme or resolves to a
            private/reserved IP address.
    """
    if _is_escape_hatch_enabled():
        logger.warning(
            "%s is enabled — skipping URL validation for: %s",
            _UNSAFE_PATHS_ENV,
            url,
        )
        return url

    parsed = urlparse(url)

    # Block file:// scheme
    if parsed.scheme == "file":
        raise ValueError(
            f"file:// URLs are not allowed: '{url}'. "
            f"Use a file path instead, or set {_UNSAFE_PATHS_ENV}=true to bypass."
        )

    # Only allow http and https
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are supported."
        )

    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: '{url}'")

    try:
        addrinfos = socket.getaddrinfo(
            parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
        )
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: '{parsed.hostname}'") from exc

    for _family, _, _, _, sockaddr in addrinfos:
        ip_str = str(sockaddr[0])
        if _is_private_or_reserved(ip_str):
            raise ValueError(
                f"URL '{url}' resolves to private/reserved IP {ip_str}. "
                f"Access to internal networks is not allowed. "
                f"Set {_UNSAFE_PATHS_ENV}=true to bypass."
            )

    return url

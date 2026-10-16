"""Shared helpers for crewai-tools tests."""

from __future__ import annotations

import os

import pytest

# Windows: creating symlinks requires an elevated prompt or Developer Mode.
_WINDOWS_SYMLINK_PRIVILEGE_WINERROR = 1314


def create_symlink_or_skip(source: str, link: str) -> None:
    """Create a symlink, skipping the test when the host lacks the privilege.

    On Windows, ``os.symlink`` raises ``OSError`` with ``winerror == 1314``
    for non-elevated users without Developer Mode. The path-containment
    regression tests must still run on hosts that can create symlinks
    (Linux, macOS, privileged Windows), so only the privilege case is skipped
    and any other error is re-raised.
    """
    try:
        os.symlink(source, link)
    except OSError as error:
        if getattr(error, "winerror", None) == _WINDOWS_SYMLINK_PRIVILEGE_WINERROR:
            pytest.skip("Creating symlinks requires elevated privileges on this Windows host")
        raise

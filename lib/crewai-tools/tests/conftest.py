"""Shared fixtures for crewai-tools tests."""

from __future__ import annotations

from collections.abc import Callable
import os

import pytest


@pytest.fixture
def symlink_or_skip() -> Callable[..., None]:
    """Return a helper that creates symlinks, skipping when the OS forbids it.

    On Windows, creating symlinks requires elevated privileges or Developer
    Mode; without them ``os.symlink`` raises ``OSError`` (WinError 1314).
    Security tests that exercise symlink handling should skip on such
    machines instead of failing.
    """

    def _symlink_or_skip(
        target: str,
        link: str,
        *,
        target_is_directory: bool = False,
    ) -> None:
        try:
            os.symlink(target, link, target_is_directory=target_is_directory)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation is not permitted on this platform: {exc}")

    return _symlink_or_skip

"""Shared fixtures for crewai-tools tests."""

from __future__ import annotations

from collections.abc import Callable
import os

import pytest


@pytest.fixture
def symlink_or_skip() -> Callable[..., None]:
    """Return a helper that creates symlinks, skipping when the OS forbids it.

    On Windows, creating symlinks requires elevated privileges or Developer
    Mode; without them ``os.symlink`` raises ``OSError`` with
    ``winerror == 1314``. Only that specific condition is skipped so unrelated
    setup failures (invalid paths, existing links, path-length limits, ...)
    stay visible.
    """

    def _symlink_or_skip(
        target: str,
        link: str,
        *,
        target_is_directory: bool = False,
    ) -> None:
        try:
            os.symlink(target, link, target_is_directory=target_is_directory)
        except OSError as exc:
            if getattr(exc, "winerror", None) != 1314:
                raise
            pytest.skip(
                "symlink creation requires elevated privileges or Developer Mode on Windows"
            )

    return _symlink_or_skip

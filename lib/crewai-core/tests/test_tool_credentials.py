"""Environment construction for private tool-repository indexes."""

from __future__ import annotations

from crewai_core.tool_credentials import build_env_with_tool_repository_credentials
import pytest


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    username: str | None = None,
    password: str | None = None,
) -> None:
    """Replace Settings with one that has the given saved credentials."""

    class _Settings:
        tool_repository_username = username
        tool_repository_password = password

    monkeypatch.setattr("crewai_core.tool_credentials.Settings", _Settings)


def test_exported_index_credentials_are_not_erased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CI job's UV_INDEX_* credentials must survive an unset setting."""
    _patch_settings(monkeypatch)
    monkeypatch.setenv("UV_INDEX_PRIVATE_USERNAME", "ci-user")
    monkeypatch.setenv("UV_INDEX_PRIVATE_PASSWORD", "ci-password")

    env = build_env_with_tool_repository_credentials("private")

    assert env["UV_INDEX_PRIVATE_USERNAME"] == "ci-user"
    assert env["UV_INDEX_PRIVATE_PASSWORD"] == "ci-password"


def test_saved_credentials_override_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_settings(monkeypatch, username="saved-user", password="saved-password")
    monkeypatch.setenv("UV_INDEX_PRIVATE_INDEX_USERNAME", "ci-user")
    monkeypatch.setenv("UV_INDEX_PRIVATE_INDEX_PASSWORD", "ci-password")

    env = build_env_with_tool_repository_credentials("private-index")

    assert env["UV_INDEX_PRIVATE_INDEX_USERNAME"] == "saved-user"
    assert env["UV_INDEX_PRIVATE_INDEX_PASSWORD"] == "saved-password"

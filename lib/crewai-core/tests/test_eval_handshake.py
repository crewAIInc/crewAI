"""`crewai eval`'s word to the run it starts: a token that counts only while
its file exists, in a folder a project's `.env` cannot move."""

from __future__ import annotations

import os
from pathlib import Path

from crewai_core import eval_handshake
import pytest


@pytest.fixture(autouse=True)
def _tokens_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        eval_handshake, "awaiting_dir", lambda: tmp_path / "eval-awaiting"
    )
    monkeypatch.delenv(eval_handshake.AWAITING_ENV, raising=False)


def test_a_run_inside_the_block_is_awaited_and_nothing_is_left_after() -> None:
    assert eval_handshake.run_awaited_by_evaluation() is False
    with eval_handshake.awaiting_evaluation() as told:
        assert told is True
        assert eval_handshake.run_awaited_by_evaluation() is True
        token = os.environ[eval_handshake.AWAITING_ENV]
        assert (eval_handshake.awaiting_dir() / token).is_file()
    assert eval_handshake.run_awaited_by_evaluation() is False
    assert eval_handshake.AWAITING_ENV not in os.environ
    assert list(eval_handshake.awaiting_dir().iterdir()) == []


@pytest.mark.parametrize("value", ["1", "true", "0" * 32, "../etc/passwd", "a" * 31])
def test_the_variable_alone_counts_for_nothing(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """What a project's `.env` could set: no file stands behind it."""
    monkeypatch.setenv(eval_handshake.AWAITING_ENV, value)
    assert eval_handshake.run_awaited_by_evaluation() is False


def test_the_value_before_the_block_is_put_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(eval_handshake.AWAITING_ENV, "an-outer-value")
    with eval_handshake.awaiting_evaluation():
        assert os.environ[eval_handshake.AWAITING_ENV] != "an-outer-value"
    assert os.environ[eval_handshake.AWAITING_ENV] == "an-outer-value"


def test_a_folder_that_cannot_be_written_tells_no_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    monkeypatch.setattr(
        eval_handshake, "awaiting_dir", lambda: blocked / "eval-awaiting"
    )
    with eval_handshake.awaiting_evaluation() as told:
        assert told is False
        assert eval_handshake.run_awaited_by_evaluation() is False


@pytest.mark.skipif(os.name == "nt", reason="the account database is POSIX")
def test_the_folder_is_the_accounts_whatever_the_environment_says(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A project's `.env` can set HOME and XDG_DATA_HOME; a token file it
    planted under them must not count."""
    monkeypatch.undo()  # the real folder, not the per-test one
    before = eval_handshake.awaiting_dir()
    monkeypatch.setenv("HOME", str(tmp_path / "planted-home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "planted-data"))

    assert eval_handshake.awaiting_dir() == before
    assert not str(eval_handshake.awaiting_dir()).startswith(str(tmp_path))


def test_the_file_carries_what_the_command_decided() -> None:
    with eval_handshake.awaiting_evaluation(
        trusted_amp_origins=["https://app.crewai.com", "https://app.crewai.com"],
        tracing_turned_on=True,
    ):
        assert eval_handshake.awaited_evaluation() == {
            "trusted_amp_origins": ["https://app.crewai.com"],
            "tracing_turned_on": True,
        }
    assert eval_handshake.awaited_evaluation() is None


@pytest.mark.parametrize(
    ("base_url", "trusted"),
    [
        ("https://app.crewai.com", True),
        ("https://APP.crewai.com/", True),
        ("https://evil.example", False),  # a `.env`'s CREWAI_PLUS_URL
        ("http://app.crewai.com", False),  # trusted, but not over plain HTTP
        ("http://localhost:3000", True),
        (None, False),
    ],
)
def test_the_saved_login_goes_only_to_an_amp_the_command_trusted(
    base_url: str | None, trusted: bool
) -> None:
    with eval_handshake.awaiting_evaluation(
        trusted_amp_origins=[
            "https://app.crewai.com",
            "http://app.crewai.com",
            "http://localhost:3000",
        ]
    ):
        assert eval_handshake.evaluation_trusts_amp(base_url) is trusted


def test_a_run_nobody_is_evaluating_is_not_this_rules_to_decide() -> None:
    assert eval_handshake.evaluation_trusts_amp("https://anywhere.example") is True


def test_an_older_commands_empty_file_is_a_request_that_trusts_no_amp() -> None:
    token = "a" * 32
    eval_handshake.awaiting_dir().mkdir(parents=True)
    (eval_handshake.awaiting_dir() / token).touch()
    os.environ[eval_handshake.AWAITING_ENV] = token
    try:
        assert eval_handshake.awaited_evaluation() == {}
        assert eval_handshake.evaluation_trusts_amp("https://app.crewai.com") is False
    finally:
        del os.environ[eval_handshake.AWAITING_ENV]

"""`crewai eval`'s word to the run it starts: a token that counts only while
its file exists."""

from __future__ import annotations

import os

from crewai_core import eval_handshake
import pytest


@pytest.fixture(autouse=True)
def _tokens_in_tmp(monkeypatch, tmp_path):
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
def test_the_variable_alone_counts_for_nothing(monkeypatch, value) -> None:
    """What a project's `.env` could set: no file stands behind it."""
    monkeypatch.setenv(eval_handshake.AWAITING_ENV, value)
    assert eval_handshake.run_awaited_by_evaluation() is False


def test_the_value_before_the_block_is_put_back(monkeypatch) -> None:
    monkeypatch.setenv(eval_handshake.AWAITING_ENV, "an-outer-value")
    with eval_handshake.awaiting_evaluation():
        assert os.environ[eval_handshake.AWAITING_ENV] != "an-outer-value"
    assert os.environ[eval_handshake.AWAITING_ENV] == "an-outer-value"


def test_a_folder_that_cannot_be_written_tells_no_run(monkeypatch, tmp_path) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("not a folder")
    monkeypatch.setattr(
        eval_handshake, "awaiting_dir", lambda: blocked / "eval-awaiting"
    )
    with eval_handshake.awaiting_evaluation() as told:
        assert told is False
        assert eval_handshake.run_awaited_by_evaluation() is False

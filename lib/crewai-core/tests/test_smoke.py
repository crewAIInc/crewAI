"""Smoke tests for the crewai-core leaf modules."""

from __future__ import annotations

import os
from pathlib import Path

from crewai_core import (
    constants,
    lock_store,
    paths,
    printer,
    user_data,
    version,
)
import pytest


def test_version_returns_string() -> None:
    v = version.get_crewai_version()
    assert isinstance(v, str) and v


def test_paths_creates_storage_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / app),
    )
    out = paths.db_storage_path()
    assert Path(out).exists()


def test_constants_exposes_env_keys() -> None:
    assert constants.CREWAI_TRAINED_AGENTS_FILE_ENV == "CREWAI_TRAINED_AGENTS_FILE"


def test_printer_emits_when_not_suppressed(capsys: pytest.CaptureFixture[str]) -> None:
    printer.PRINTER.print("hello", color="green")
    out = capsys.readouterr().out
    assert "hello" in out


def test_printer_respects_suppression(capsys: pytest.CaptureFixture[str]) -> None:
    token = printer.set_suppress_console_output(True)
    try:
        printer.PRINTER.print("hidden")
    finally:
        printer._suppress_console_output.reset(token)  # type: ignore[arg-type]
    assert "hidden" not in capsys.readouterr().out


def test_lock_acquires_and_releases() -> None:
    with lock_store.lock("crewai_core.tests.smoke", timeout=5):
        pass


def test_user_data_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", "crewai_core_test_user_data")
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / app),
    )
    user_data.update_user_data({"trace_consent": True, "first_execution_done": True})
    data = user_data._load_user_data()
    assert data == {"trace_consent": True, "first_execution_done": True}
    assert user_data.has_user_declined_tracing() is False
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    assert user_data.is_tracing_enabled() is True
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    assert (
        user_data.is_tracing_enabled() is True
    )  # consent alone enables (matches runtime)


def test_user_data_decline_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", "crewai_core_test_decline")
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / app),
    )
    user_data.update_user_data({"trace_consent": False, "first_execution_done": True})
    assert user_data.has_user_declined_tracing() is True
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    assert user_data.is_tracing_enabled() is False
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    assert user_data.is_tracing_enabled() is True  # env-var override (matches runtime)


def test_unused_var_warning_silenced() -> None:
    # Touch os to keep the import (used by env-var fixtures above)
    assert os.environ is not None

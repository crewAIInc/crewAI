"""Tests for CrewAI Core settings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from typing import Any

import crewai_core.settings as settings_module
import pytest


def test_concurrent_settings_instances_keep_configured_enterprise_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent settings probes must not make a worker use fallback settings."""
    config_path = tmp_path / "config" / "settings.json"
    config_path.parent.mkdir()
    enterprise_url = "https://enterprise.example.com"
    config_path.write_text(json.dumps({"enterprise_base_url": enterprise_url}))
    monkeypatch.setattr(settings_module, "DEFAULT_CONFIG_PATH", config_path)

    original_unlink = Path.unlink
    shared_probe_barrier = threading.Barrier(2)

    def synchronize_shared_probe_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.name == ".crewai_write_test":
            shared_probe_barrier.wait(timeout=1)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", synchronize_shared_probe_unlink)

    with ThreadPoolExecutor(max_workers=2) as executor:
        settings = list(executor.map(lambda _: settings_module.Settings(), range(2)))

    assert [item.enterprise_base_url for item in settings] == [enterprise_url] * 2
    assert not list(config_path.parent.glob(".crewai_write_test.*"))


@pytest.mark.parametrize(
    "contents",
    ['{"org_name": "acme",}', "not json", "[]"],
    ids=["trailing-comma", "garbage", "non-object"],
)
def test_corrupt_settings_file_is_reported_and_replaced_on_save(
    tmp_path: Path, contents: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt settings file is logged on load and backed up by the next save."""
    config_path = tmp_path / "settings.json"
    config_path.write_text(contents)

    with caplog.at_level("WARNING", logger=settings_module.__name__):
        settings = settings_module.Settings(config_path=config_path)
    assert settings.org_name is None
    assert str(config_path) in caplog.text

    settings.org_name = "new-org"
    settings.dump()

    assert json.loads(config_path.read_text())["org_name"] == "new-org"
    backups = list(tmp_path.glob("settings.json.*.bak"))
    assert [b.read_text() for b in backups] == [contents]


def test_each_corruption_gets_its_own_backup(tmp_path: Path) -> None:
    """A second corrupt file does not overwrite the backup of the first."""
    config_path = tmp_path / "settings.json"
    for contents in ('{"org_name": "first",}', '{"org_name": "second",}'):
        config_path.write_text(contents)
        settings_module.Settings(config_path=config_path).dump()

    backups = sorted(b.read_text() for b in tmp_path.glob("settings.json.*.bak"))
    assert backups == ['{"org_name": "first",}', '{"org_name": "second",}']


def test_settings_file_that_cannot_be_read_is_left_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid settings file that can't be read is never moved aside or replaced."""
    config_path = tmp_path / "settings.json"
    original = '{"org_name": "acme", "tool_repository_username": "u1"}'
    config_path.write_text(original)
    real_open = Path.open

    def deny_reading_settings(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> Any:
        if self == config_path and "r" in mode:
            raise PermissionError("simulated: file locked by another process")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_reading_settings)
    settings = settings_module.Settings(config_path=config_path)
    settings.org_name = "new-org"
    settings.dump()
    monkeypatch.undo()

    assert config_path.read_text() == original
    assert not list(tmp_path.glob("settings.json.*.bak"))

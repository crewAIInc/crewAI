"""Tests for CrewAI Core settings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

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

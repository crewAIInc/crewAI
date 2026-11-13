"""Tests for checkpoint provider detection."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from crewai.state.provider.json_provider import JsonProvider
from crewai.state.provider.sqlite_provider import SqliteProvider
from crewai.state.provider.utils import detect_provider


class TestDetectProvider:
    def test_sqlite_location_with_checkpoint_id(self, tmp_path: Path) -> None:
        db = tmp_path / "cp.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")

        provider = detect_provider(f"{db}#abc123")
        assert isinstance(provider, SqliteProvider)

    def test_sqlite_location_with_hash_in_directory(self, tmp_path: Path) -> None:
        # A directory may legitimately contain "#"; the checkpoint id always
        # follows the *last* "#", so the db path is everything before it.
        run_dir = tmp_path / "run#1"
        run_dir.mkdir()
        db = run_dir / "cp.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")

        provider = detect_provider(f"{db}#abc123")
        assert isinstance(provider, SqliteProvider)

    def test_multiple_checkpoints_in_hash_directory(
        self, tmp_path: Path
    ) -> None:
        # Every "#" but the last belongs to the db path, so each checkpoint of
        # the same db resolves to the same file.
        run_dir = tmp_path / "run#1"
        run_dir.mkdir()
        db = run_dir / "cp.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")

        for checkpoint_id in ("abc", "def"):
            provider = detect_provider(f"{db}#{checkpoint_id}")
            assert isinstance(provider, SqliteProvider)

    def test_json_file_falls_back_to_json_provider(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "cp.json"
        checkpoint.write_text(json.dumps({"state": {}}))

        provider = detect_provider(f"{checkpoint}#abc123")
        assert isinstance(provider, JsonProvider)

    def test_missing_file_falls_back_to_json_provider(self, tmp_path: Path) -> None:
        provider = detect_provider(str(tmp_path / "missing.db") + "#abc123")
        assert isinstance(provider, JsonProvider)

    def test_plain_directory(self, tmp_path: Path) -> None:
        provider = detect_provider(str(tmp_path))
        assert isinstance(provider, JsonProvider)

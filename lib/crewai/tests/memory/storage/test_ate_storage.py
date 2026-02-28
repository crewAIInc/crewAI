"""Tests for AteStorage — FoodforThought .mv2 memory backend.

All subprocess calls are mocked; no real ``ate`` CLI invocations occur.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from crewai.memory.storage.ate_storage import AteStorage


def _ok(stdout: str = "", stderr: str = "") -> MagicMock:
    """Helper — build a CompletedProcess-like mock for returncode 0."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = stderr
    return m


def _fail(stderr: str = "error") -> MagicMock:
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = stderr
    return m


# ── Construction ────────────────────────────────────────────────────


class TestConstruction(unittest.TestCase):
    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def test_valid_type(self, _w: Any) -> None:
        for t in ("short_term", "long_term", "entities", "external"):
            s = AteStorage(type=t)
            self.assertEqual(s.type, t)

    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def test_invalid_type_raises(self, _w: Any) -> None:
        with self.assertRaises(ValueError):
            AteStorage(type="invalid")

    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value=None)
    def test_missing_cli_raises(self, _w: Any) -> None:
        with self.assertRaises(FileNotFoundError):
            AteStorage(type="long_term")


# ── Default paths ───────────────────────────────────────────────────


class TestDefaultPaths(unittest.TestCase):
    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def test_default_path_no_crew(self, _w: Any) -> None:
        s = AteStorage(type="long_term")
        expected = str(Path.home() / ".ate/memories/default/long_term.mv2")
        self.assertEqual(s._memory_path, expected)

    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def test_custom_path_from_config(self, _w: Any) -> None:
        s = AteStorage(type="short_term", config={"memory_path": "/tmp/my.mv2"})
        self.assertEqual(s._memory_path, "/tmp/my.mv2")

    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def test_crew_name_in_path(self, _w: Any) -> None:
        crew = MagicMock()
        crew.name = "Research Team"
        s = AteStorage(type="entities", crew=crew)
        self.assertIn("research_team", s._memory_path)

    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def test_each_type_gets_own_file(self, _w: Any) -> None:
        paths = set()
        for t in ("short_term", "long_term", "entities", "external"):
            s = AteStorage(type=t)
            paths.add(s._memory_path)
        self.assertEqual(len(paths), 4)


# ── Save ────────────────────────────────────────────────────────────


class TestSave(unittest.TestCase):
    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def setUp(self, _w: Any) -> None:
        self.storage = AteStorage(type="long_term")

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_save_calls_ate(self, _e: Any, mock_run: Any) -> None:
        self.storage.save("hello world", {"agent": "researcher"})
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "ate")
        self.assertIn("--text", args)
        self.assertIn("hello world", args)

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_save_includes_type_tag(self, _e: Any, mock_run: Any) -> None:
        self.storage.save("data", {"agent": "a"})
        args = mock_run.call_args[0][0]
        tags_idx = args.index("--tags") + 1
        self.assertIn("long_term", args[tags_idx])

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_save_appends_custom_tags(self, _e: Any, mock_run: Any) -> None:
        self.storage.save("data", {"agent": "a", "tags": ["custom", "extra"]})
        args = mock_run.call_args[0][0]
        tags_idx = args.index("--tags") + 1
        self.assertIn("custom", args[tags_idx])


# ── Search ──────────────────────────────────────────────────────────


class TestSearch(unittest.TestCase):
    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def setUp(self, _w: Any) -> None:
        self.storage = AteStorage(type="long_term")

    @patch("crewai.memory.storage.ate_storage.subprocess.run")
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_search_returns_formatted(self, _e: Any, mock_run: Any) -> None:
        mock_run.return_value = _ok(json.dumps({
            "results": [{"text": "memory content", "score": 0.9, "tags": ["long_term"]}]
        }))
        results = self.storage.search("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "memory content")
        self.assertEqual(results[0]["score"], 0.9)

    @patch("crewai.memory.storage.ate_storage.subprocess.run")
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_search_filters_by_threshold(self, _e: Any, mock_run: Any) -> None:
        mock_run.return_value = _ok(json.dumps({
            "results": [
                {"text": "high", "score": 0.9},
                {"text": "low", "score": 0.1},
            ]
        }))
        results = self.storage.search("q", score_threshold=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "high")

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_fail())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_search_returns_empty_on_error(self, _e: Any, _r: Any) -> None:
        results = self.storage.search("query")
        self.assertEqual(results, [])

    @patch("crewai.memory.storage.ate_storage.subprocess.run")
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_search_handles_bare_list(self, _e: Any, mock_run: Any) -> None:
        mock_run.return_value = _ok(json.dumps([
            {"text": "item", "score": 0.8}
        ]))
        results = self.storage.search("q")
        self.assertEqual(len(results), 1)

    @patch("crewai.memory.storage.ate_storage.subprocess.run")
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_search_passes_limit(self, _e: Any, mock_run: Any) -> None:
        mock_run.return_value = _ok("[]")
        self.storage.search("q", limit=3)
        args = mock_run.call_args[0][0]
        self.assertIn("3", args)


# ── Reset ───────────────────────────────────────────────────────────


class TestReset(unittest.TestCase):
    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def setUp(self, _w: Any) -> None:
        self.storage = AteStorage(type="long_term")

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    @patch("crewai.memory.storage.ate_storage.Path.unlink")
    @patch("crewai.memory.storage.ate_storage.Path.mkdir")
    def test_reset_reinits(self, _mk: Any, _u: Any, _e: Any, mock_run: Any) -> None:
        self.storage.reset()
        args = mock_run.call_args[0][0]
        self.assertIn("init", args)


# ── Auto-init ───────────────────────────────────────────────────────


class TestAutoInit(unittest.TestCase):
    @patch("crewai.memory.storage.ate_storage.shutil.which", return_value="/usr/bin/ate")
    def setUp(self, _w: Any) -> None:
        self.storage = AteStorage(type="short_term")

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=False)
    @patch("crewai.memory.storage.ate_storage.Path.mkdir")
    def test_auto_init_on_save(self, _mk: Any, _e: Any, mock_run: Any) -> None:
        self.storage.save("data", {})
        calls = [c[0][0] for c in mock_run.call_args_list]
        init_calls = [c for c in calls if "init" in c]
        self.assertTrue(len(init_calls) > 0)

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok("[]"))
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=False)
    @patch("crewai.memory.storage.ate_storage.Path.mkdir")
    def test_auto_init_on_search(self, _mk: Any, _e: Any, mock_run: Any) -> None:
        self.storage.search("query")
        calls = [c[0][0] for c in mock_run.call_args_list]
        init_calls = [c for c in calls if "init" in c]
        self.assertTrue(len(init_calls) > 0)

    @patch("crewai.memory.storage.ate_storage.subprocess.run", return_value=_ok())
    @patch("crewai.memory.storage.ate_storage.Path.exists", return_value=True)
    def test_skips_init_when_exists(self, _e: Any, mock_run: Any) -> None:
        self.storage.save("data", {})
        calls = [c[0][0] for c in mock_run.call_args_list]
        init_calls = [c for c in calls if "init" in c]
        self.assertEqual(len(init_calls), 0)


if __name__ == "__main__":
    unittest.main()

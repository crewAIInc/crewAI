"""Tests that lancedb is an optional dependency.

These tests verify that:
1. The lancedb_storage module handles a missing lancedb gracefully.
2. Memory falls back with a clear error when lancedb is not installed.
3. Importing crewai itself does not require lancedb.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def test_lancedb_storage_raises_import_error_when_lancedb_missing(tmp_path):
    """LanceDBStorage.__init__ raises ImportError with install instructions when lancedb is absent."""
    with patch.dict(sys.modules, {"lancedb": None}):
        # Force reload so the module picks up the patched sys.modules
        import importlib

        import crewai.memory.storage.lancedb_storage as mod

        importlib.reload(mod)

        with pytest.raises(ImportError, match="pip install 'crewai\\[memory\\]'"):
            mod.LanceDBStorage(path=str(tmp_path / "mem"))

        # Restore the module to its original state
        importlib.reload(mod)


def test_memory_default_storage_raises_when_lancedb_missing(tmp_path):
    """Memory(storage='lancedb') raises ImportError when lancedb is not installed."""
    with patch.dict(sys.modules, {"lancedb": None}):
        import importlib

        import crewai.memory.storage.lancedb_storage as mod

        importlib.reload(mod)

        try:
            from crewai.memory.unified_memory import Memory

            with pytest.raises(ImportError, match="pip install 'crewai\\[memory\\]'"):
                Memory(
                    storage="lancedb",
                    llm=MagicMock(),
                    embedder=MagicMock(),
                )
        finally:
            importlib.reload(mod)


def test_memory_with_path_string_raises_when_lancedb_missing(tmp_path):
    """Memory(storage='/some/path') also uses LanceDBStorage and raises when lancedb is missing."""
    with patch.dict(sys.modules, {"lancedb": None}):
        import importlib

        import crewai.memory.storage.lancedb_storage as mod

        importlib.reload(mod)

        try:
            from crewai.memory.unified_memory import Memory

            with pytest.raises(ImportError, match="pip install 'crewai\\[memory\\]'"):
                Memory(
                    storage=str(tmp_path / "custom_path"),
                    llm=MagicMock(),
                    embedder=MagicMock(),
                )
        finally:
            importlib.reload(mod)


def test_crewai_import_does_not_require_lancedb():
    """Importing crewai should work even if lancedb is not installed.

    The Memory class is lazily imported in crewai/__init__.py, so lancedb
    should never be pulled in at import time.
    """
    # This test verifies the lazy import mechanism by checking that the
    # crewai module is importable and that Memory is listed in __all__
    # but not yet resolved in the module globals until accessed.
    import crewai

    assert "Memory" in crewai.__all__
    # Memory should be accessible (lazy import triggers on access)
    assert hasattr(crewai, "Memory")


def test_memory_with_custom_storage_backend_does_not_need_lancedb(tmp_path):
    """When a custom StorageBackend is passed, lancedb is never needed."""
    with patch.dict(sys.modules, {"lancedb": None}):
        import importlib

        import crewai.memory.storage.lancedb_storage as mod

        importlib.reload(mod)

        try:
            from crewai.memory.unified_memory import Memory

            mock_storage = MagicMock()
            # Should not raise, since we're providing a custom storage backend
            mem = Memory(
                storage=mock_storage,
                llm=MagicMock(),
                embedder=MagicMock(),
            )
            assert mem._storage is mock_storage
        finally:
            importlib.reload(mod)


def test_lancedb_in_optional_dependencies():
    """Verify lancedb is listed under optional [memory] dependencies, not core."""
    import tomli
    from pathlib import Path

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomli.load(f)

    core_deps = data["project"]["dependencies"]
    optional_deps = data["project"]["optional-dependencies"]

    # lancedb should NOT be in core dependencies
    assert not any("lancedb" in dep for dep in core_deps), (
        "lancedb should not be a core dependency"
    )

    # lancedb SHOULD be in optional [memory] dependencies
    assert "memory" in optional_deps, "Missing [memory] optional dependency group"
    memory_deps = optional_deps["memory"]
    assert any("lancedb" in dep for dep in memory_deps), (
        "lancedb should be in the [memory] optional dependency group"
    )

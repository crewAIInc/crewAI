"""Regression tests for ChromaDB as an optional dependency."""

import subprocess
import sys


def test_crewai_imports_without_chromadb() -> None:
    """CrewAI's base import must not require the optional ChromaDB package."""
    code = """
import importlib.abc
import sys

class BlockChromaDB(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'chromadb' or fullname.startswith('chromadb.'):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockChromaDB())
import crewai
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

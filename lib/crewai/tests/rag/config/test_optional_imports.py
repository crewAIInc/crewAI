"""Tests for optional imports."""

import subprocess
import sys

import pytest
from crewai.rag.config.optional_imports.base import _MissingProvider
from crewai.rag.config.optional_imports.providers import (
    MissingChromaDBConfig,
    MissingMilvusConfig,
)


def test_missing_provider_raises_runtime_error():
    """Test that _MissingProvider raises RuntimeError on instantiation."""
    with pytest.raises(
        RuntimeError, match="provider '__missing__' requested but not installed"
    ):
        _MissingProvider()


def test_missing_chromadb_config_raises_runtime_error():
    """Test that MissingChromaDBConfig raises RuntimeError on instantiation."""
    with pytest.raises(
        RuntimeError, match="provider 'chromadb' requested but not installed"
    ):
        MissingChromaDBConfig()


def test_missing_milvus_config_raises_runtime_error():
    """Test that MissingMilvusConfig raises RuntimeError on instantiation."""
    with pytest.raises(
        RuntimeError,
        match=r'Install the extra: `uv add "crewai\[milvus\]"`\.',
    ):
        MissingMilvusConfig()


def test_config_types_falls_back_when_pymilvus_is_missing() -> None:
    """Test that config types use the missing-provider contract without pymilvus."""
    script = """
import importlib.abc
import sys


class BlockPymilvus(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "pymilvus" or fullname.startswith("pymilvus."):
            raise ModuleNotFoundError("blocked pymilvus for optional import test")
        return None


sys.meta_path.insert(0, BlockPymilvus())

from crewai.rag.config.optional_imports.providers import MissingMilvusConfig
from crewai.rag.config.types import MilvusConfig

assert MilvusConfig is MissingMilvusConfig

try:
    MilvusConfig()
except RuntimeError as error:
    assert 'uv add "crewai[milvus]"' in str(error)
else:
    raise AssertionError("MissingMilvusConfig did not raise RuntimeError")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

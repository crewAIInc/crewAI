"""Tests for optional imports."""

import pytest
from crewai.rag.config.optional_imports.base import _MissingProvider
from crewai.rag.config.optional_imports.providers import MissingChromaDBConfig


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

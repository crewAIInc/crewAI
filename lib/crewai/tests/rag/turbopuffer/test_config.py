"""Tests for turbopuffer configuration defaults."""

import builtins

import pytest
from crewai.rag.turbopuffer.config import _default_embedding_function


def test_default_embedding_function_defers_fastembed(monkeypatch):
    """The default embedder must not import fastembed or load the model until the
    first embedding call, so config construction never triggers a download."""
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "fastembed" or name.startswith("fastembed."):
            raise ImportError("fastembed blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    # Building the default must not import fastembed / download anything.
    embed_fn = _default_embedding_function()
    assert callable(embed_fn)

    # fastembed is only needed on the first call; a missing optional dependency
    # surfaces as a clear, actionable error there.
    with pytest.raises(ImportError, match="fastembed"):
        embed_fn("hello world")

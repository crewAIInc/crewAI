"""Test that LanceDB adapter has proper docstrings."""

import inspect

import pytest

lancedb = pytest.importorskip("lancedb")

from crewai_tools.adapters.lancedb_adapter import (
    LanceDBAdapter,
    _default_embedding_function,
)


def test_lancedb_adapter_class_has_docstring():
    """Verify that LanceDBAdapter class has a docstring."""
    assert LanceDBAdapter.__doc__ is not None, "LanceDBAdapter class is missing a docstring"
    assert len(LanceDBAdapter.__doc__.strip()) > 0, "LanceDBAdapter docstring is empty"


def test_lancedb_adapter_model_post_init_has_docstring():
    """Verify that model_post_init method has a docstring."""
    assert (
        LanceDBAdapter.model_post_init.__doc__ is not None
    ), "model_post_init method is missing a docstring"
    assert (
        len(LanceDBAdapter.model_post_init.__doc__.strip()) > 0
    ), "model_post_init docstring is empty"


def test_lancedb_adapter_query_has_docstring():
    """Verify that query method has a docstring."""
    assert LanceDBAdapter.query.__doc__ is not None, "query method is missing a docstring"
    assert len(LanceDBAdapter.query.__doc__.strip()) > 0, "query docstring is empty"


def test_lancedb_adapter_add_has_docstring():
    """Verify that add method has a docstring."""
    assert LanceDBAdapter.add.__doc__ is not None, "add method is missing a docstring"
    assert len(LanceDBAdapter.add.__doc__.strip()) > 0, "add docstring is empty"


def test_default_embedding_function_has_docstring():
    """Verify that _default_embedding_function has a docstring."""
    assert (
        _default_embedding_function.__doc__ is not None
    ), "_default_embedding_function is missing a docstring"
    assert (
        len(_default_embedding_function.__doc__.strip()) > 0
    ), "_default_embedding_function docstring is empty"


def test_docstrings_contain_required_sections():
    """Verify that docstrings contain Args, Returns, or Example sections where appropriate."""
    query_doc = LanceDBAdapter.query.__doc__
    assert query_doc is not None
    assert "Args:" in query_doc or "Parameters:" in query_doc, "query docstring should have Args/Parameters section"
    assert "Returns:" in query_doc, "query docstring should have Returns section"

    add_doc = LanceDBAdapter.add.__doc__
    assert add_doc is not None
    assert "Args:" in add_doc or "Parameters:" in add_doc, "add docstring should have Args/Parameters section"

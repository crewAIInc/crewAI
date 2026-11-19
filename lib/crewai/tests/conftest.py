"""Shared test configuration for CrewAI's core package."""

from collections.abc import Generator
from importlib.util import find_spec

import pytest
from crewai.rag.config.utils import clear_rag_config


@pytest.fixture(autouse=True)
def reset_global_rag_client() -> Generator[None, None, None]:
    """Drop the cached global RAG client before and after every test.

    It lives in a context variable; without a reset, a client built by one test
    (real storage, real embedder) would bypass the patches of the next test that
    runs in the same worker.
    """
    clear_rag_config()
    yield
    clear_rag_config()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``requires_chromadb`` tests when chromadb is not installed."""
    if find_spec("chromadb") is not None:
        return
    skip = pytest.mark.skip(reason="chromadb is not installed (crewai[chromadb])")
    for item in items:
        if "requires_chromadb" in item.keywords:
            item.add_marker(skip)

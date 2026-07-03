"""Tests for the SemanticToolFilter (embedding-based MCP tool filtering)."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from crewai.mcp import (
    SemanticToolFilter,
    ToolFilterContext,
    create_semantic_tool_filter,
)


VOCAB = ["search", "web", "file", "read", "code", "execute", "database", "query"]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class BagOfWordsEmbedder:
    """Deterministic multi-hot embedder over a fixed vocabulary.

    Cosine similarity of two vectors equals the fraction of shared vocab words
    scaled by each vector's magnitude — realistic enough to exercise the filter
    without any model or network. Tracks call count to verify caching.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0 if word in _tokens(text) else 0.0 for word in VOCAB] for text in input]


class RaisingEmbedder:
    """Embedder that always raises, to exercise fail-open behavior."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        raise RuntimeError("embedder unavailable")


def _agent(role: str = "", goal: str = "", backstory: str = "") -> Any:
    agent = MagicMock()
    agent.role = role
    agent.goal = goal
    agent.backstory = backstory
    return agent


class TestSemanticToolFilter:
    """Tests for SemanticToolFilter."""

    def test_includes_relevant_excludes_irrelevant(self) -> None:
        """Tools semantically close to the agent profile are included."""
        embedder = BagOfWordsEmbedder()
        flt = SemanticToolFilter(embedder=embedder, threshold=0.3)
        ctx = ToolFilterContext(
            agent=_agent(role="Researcher", goal="search the web"),
            server_name="srv",
            run_context=None,
        )

        web_search = {"name": "web_search", "description": "search the web"}
        read_file = {"name": "read_file", "description": "read a file from disk"}

        assert flt(ctx, web_search) is True
        assert flt(ctx, read_file) is False

    def test_fail_open_when_no_query(self) -> None:
        """With no agent and no run_context query, all tools are included."""
        embedder = BagOfWordsEmbedder()
        flt = SemanticToolFilter(embedder=embedder, threshold=0.3)
        ctx = ToolFilterContext(agent=None, server_name="srv", run_context=None)

        assert flt(ctx, {"name": "any", "description": "anything"}) is True
        assert embedder.calls == 0

    def test_fail_open_when_embedder_raises(self) -> None:
        """Embedder failures never strip tools."""
        flt = SemanticToolFilter(embedder=RaisingEmbedder(), threshold=0.3)
        ctx = ToolFilterContext(
            agent=_agent(role="Researcher", goal="search the web"),
            server_name="srv",
            run_context=None,
        )
        assert flt(ctx, {"name": "web_search", "description": "search the web"}) is True

    def test_run_context_query_takes_precedence(self) -> None:
        """An explicit run_context query overrides the agent profile."""
        embedder = BagOfWordsEmbedder()
        flt = SemanticToolFilter(embedder=embedder, threshold=0.3)
        ctx = ToolFilterContext(
            agent=_agent(role="Coder", goal="write code"),
            server_name="srv",
            run_context={"query": "search the web"},
        )
        # Relevant to the run_context query, not the coder profile.
        assert flt(ctx, {"name": "web_search", "description": "search the web"}) is True
        # Relevant to the coder profile but not the query.
        assert flt(ctx, {"name": "run_code", "description": "execute code"}) is False

    def test_threshold_boundary(self) -> None:
        """threshold controls the include/exclude cutoff."""
        # query "search the web" -> {search, web} (magnitude sqrt(2))
        # tool "search" -> {search} -> cosine = 1 / (sqrt(2) * 1) ~= 0.707
        embedder = BagOfWordsEmbedder()
        ctx = ToolFilterContext(
            agent=_agent(role="R", goal="search the web"),
            server_name="srv",
            run_context=None,
        )
        flt_low = SemanticToolFilter(embedder=BagOfWordsEmbedder(), threshold=0.5)
        flt_high = SemanticToolFilter(embedder=BagOfWordsEmbedder(), threshold=0.8)
        tool = {"name": "search_only", "description": "search"}

        assert flt_low(ctx, tool) is True  # 0.707 >= 0.5
        assert flt_high(ctx, tool) is False  # 0.707 < 0.8

    def test_embeddings_are_cached(self) -> None:
        """The query is embedded once even when filtering many tools."""
        embedder = BagOfWordsEmbedder()
        flt = SemanticToolFilter(embedder=embedder, threshold=0.3)
        ctx = ToolFilterContext(
            agent=_agent(role="Researcher", goal="search the web"),
            server_name="srv",
            run_context=None,
        )
        flt(ctx, {"name": "web_search", "description": "search the web"})
        flt(ctx, {"name": "read_file", "description": "read a file"})

        # 1 query + 2 distinct tool texts = 3 embedder calls (no re-embed of query).
        assert embedder.calls == 3

    def test_factory_returns_working_filter(self) -> None:
        """create_semantic_tool_filter returns a usable SemanticToolFilter."""
        flt = create_semantic_tool_filter(
            embedder=BagOfWordsEmbedder(), threshold=0.3
        )
        assert isinstance(flt, SemanticToolFilter)
        ctx = ToolFilterContext(
            agent=_agent(role="Researcher", goal="search the web"),
            server_name="srv",
            run_context=None,
        )
        assert flt(ctx, {"name": "web_search", "description": "search the web"}) is True
        assert flt(ctx, {"name": "read_file", "description": "read a file"}) is False

    def test_empty_tool_description_fail_open(self) -> None:
        """A tool with no name/description is included (nothing to embed)."""
        flt = SemanticToolFilter(embedder=BagOfWordsEmbedder(), threshold=0.3)
        ctx = ToolFilterContext(
            agent=_agent(role="Researcher", goal="search the web"),
            server_name="srv",
            run_context=None,
        )
        assert flt(ctx, {"name": "", "description": ""}) is True

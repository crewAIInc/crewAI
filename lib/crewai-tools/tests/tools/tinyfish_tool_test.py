"""Tests for the TinyFish tool family."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.tinyfish_tool import base as tf_base
from crewai_tools.tools.tinyfish_tool.schemas import (
    TinyfishAgentParams,
    TinyfishFetchParams,
)
from crewai_tools.tools.tinyfish_tool.tinyfish_agent_tool import TinyfishAgentTool
from crewai_tools.tools.tinyfish_tool.tinyfish_fetch_tool import TinyfishFetchTool
from crewai_tools.tools.tinyfish_tool.tinyfish_search_tool import TinyfishSearchTool

_TOOL_CLASSES = (TinyfishAgentTool, TinyfishSearchTool, TinyfishFetchTool)


def _response(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(model_dump=lambda: payload)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    monkeypatch.delenv("TF_API_INTEGRATION", raising=False)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _TOOL_CLASSES)
def test_tool_declares_required_env_var(cls: type[TinyfishAgentTool]) -> None:
    env_vars = cls().env_vars
    assert [e.name for e in env_vars] == ["TINYFISH_API_KEY"]
    assert env_vars[0].required is True


@pytest.mark.parametrize("cls", _TOOL_CLASSES)
def test_tool_declares_package_dependency(cls: type[TinyfishAgentTool]) -> None:
    assert cls().package_dependencies == ["tinyfish"]


@pytest.mark.parametrize(
    "cls,expected_name",
    [
        (TinyfishAgentTool, "Tinyfish Agent"),
        (TinyfishSearchTool, "Tinyfish Search"),
        (TinyfishFetchTool, "Tinyfish Fetch"),
    ],
)
def test_tool_humanized_name(
    cls: type[TinyfishAgentTool], expected_name: str
) -> None:
    assert cls().name == expected_name


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_api_key_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TINYFISH_API_KEY", raising=False)
    result = TinyfishSearchTool().run(query="hello")
    assert "TINYFISH_API_KEY is not set" in result


def test_missing_sdk_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf_base, "_TINYFISH_AVAILABLE", False)
    monkeypatch.setattr(
        tf_base, "_TINYFISH_IMPORT_ERROR", "No module named 'tinyfish'"
    )
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")

    result = TinyfishSearchTool().run(query="hello")

    assert "'tinyfish' Python SDK is not installed" in result
    assert "No module named 'tinyfish'" in result


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_agent_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        TinyfishAgentParams(url="ftp://example.com", goal="x")


def test_fetch_rejects_empty_urls() -> None:
    with pytest.raises(ValueError):
        TinyfishFetchParams(urls=[])


def test_fetch_rejects_too_many_urls() -> None:
    with pytest.raises(ValueError):
        TinyfishFetchParams(urls=[f"https://e.com/{i}" for i in range(11)])


# ---------------------------------------------------------------------------
# Happy paths (mocked SDK)
# ---------------------------------------------------------------------------


def test_agent_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")

    fake_client = MagicMock()
    fake_client.agent.run.return_value = _response(
        {"status": "COMPLETED", "result": {"title": "Hello"}}
    )

    with patch.object(tf_base, "TinyFish", return_value=fake_client) as ctor:
        out = TinyfishAgentTool().run(
            url="https://example.com",
            goal="Get the title",
            browser_profile="lite",
        )

    ctor.assert_called_once_with(api_key="sk-test")
    fake_client.agent.run.assert_called_once()
    assert json.loads(out) == {"status": "COMPLETED", "result": {"title": "Hello"}}


def test_search_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")

    fake_client = MagicMock()
    fake_client.search.query.return_value = _response(
        {"query": "hello", "results": [], "total_results": 0}
    )

    with patch.object(tf_base, "TinyFish", return_value=fake_client):
        out = TinyfishSearchTool().run(query="hello", language="en")

    fake_client.search.query.assert_called_once_with(
        query="hello", location=None, language="en"
    )
    assert json.loads(out)["query"] == "hello"


def test_fetch_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")

    fake_client = MagicMock()
    fake_client.fetch.get_contents.return_value = _response(
        {"results": [{"url": "https://example.com", "text": "hi"}]}
    )

    with patch.object(tf_base, "TinyFish", return_value=fake_client):
        out = TinyfishFetchTool().run(urls=["https://example.com"], format="markdown")

    fake_client.fetch.get_contents.assert_called_once_with(
        urls=["https://example.com"],
        format="markdown",
        links=False,
        image_links=False,
    )
    assert "example.com" in out


def test_integration_tag_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")
    fake_client = MagicMock()
    fake_client.search.query.return_value = _response({"results": []})

    with patch.object(tf_base, "TinyFish", return_value=fake_client):
        TinyfishSearchTool().run(query="hi")

    import os

    assert os.environ.get("TF_API_INTEGRATION") == "crewai-tools"


def test_sdk_exception_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")
    fake_client = MagicMock()
    fake_client.search.query.side_effect = RuntimeError("boom")

    with patch.object(tf_base, "TinyFish", return_value=fake_client):
        result = TinyfishSearchTool().run(query="hi")

    assert result.startswith("Error: RuntimeError: boom")


def test_client_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TINYFISH_API_KEY", "sk-test")
    fake_client = MagicMock()
    fake_client.search.query.return_value = _response({"results": []})

    with patch.object(tf_base, "TinyFish", return_value=fake_client) as ctor:
        tool = TinyfishSearchTool()
        tool.run(query="one")
        tool.run(query="two")

    assert ctor.call_count == 1

from collections.abc import Iterator
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai_tools import ArcmiraSearchTool
import pytest
import requests


@pytest.fixture
def api_response() -> Iterator[tuple[MagicMock, MagicMock]]:
    with patch(
        "crewai_tools.tools.arcmira_search_tool.arcmira_search_tool.requests.get"
    ) as get:
        response = MagicMock(status_code=200)
        get.return_value = response
        yield get, response


def test_search_preserves_evidence_and_coverage(
    api_response: tuple[MagicMock, MagicMock],
) -> None:
    get, response = api_response
    response.json.return_value = {
        "query": "open source",
        "returned": 1,
        "chunks": [
            {
                "text": "We released the project as open source.",
                "video_id": "example0001",
                "start_seconds": 12,
                "watch_url": "https://www.youtube.com/watch?v=example0001&t=12s",
            }
        ],
        "partial": True,
        "failed_batches": 1,
        "search_index": {"state": "catching_up", "missing_before": "2026-07-01"},
        "access": {"code": "freshness_requires_paid"},
    }
    tool = ArcmiraSearchTool(api_key="test-key")
    result = tool.run(query="open source", limit=3)
    assert result == {
        "query": "open source",
        "returned": 1,
        "chunks": [
            {
                "text": "We released the project as open source.",
                "video_id": "example0001",
                "start_seconds": 12,
                "watch_url": "https://www.youtube.com/watch?v=example0001&t=12s",
            }
        ],
        "partial": True,
        "failed_batches": 1,
        "search_index": {"state": "catching_up", "missing_before": "2026-07-01"},
        "access": {"code": "freshness_requires_paid"},
    }
    get.assert_called_once_with(
        "https://api.arcmira.com/v1/search",
        params={"q": "open source", "limit": 3},
        headers={"Authorization": "Bearer test-key"},
        timeout=30,
        allow_redirects=False,
    )


def test_environment_key_stays_out_of_model_input_and_serialization(
    monkeypatch: pytest.MonkeyPatch, api_response: tuple[MagicMock, MagicMock]
) -> None:
    monkeypatch.setenv("ARCMIRA_API_KEY", "environment-key")
    get, response = api_response
    response.json.return_value = {"query": "open source", "chunks": [], "returned": 0}
    tool = ArcmiraSearchTool()
    assert tool.run(query="open source") == {
        "query": "open source",
        "chunks": [],
        "returned": 0,
    }
    assert get.call_args.kwargs["headers"] == {
        "Authorization": "Bearer environment-key"
    }
    assert set(tool.args_schema.model_json_schema()["properties"]) == {"query", "limit"}
    assert "environment-key" not in tool.model_dump_json()
    assert "environment-key" not in repr(tool)
    assert "api_key" not in tool.model_dump()


def test_explicit_key_overrides_environment(
    monkeypatch: pytest.MonkeyPatch, api_response: tuple[MagicMock, MagicMock]
) -> None:
    monkeypatch.setenv("ARCMIRA_API_KEY", "environment-key")
    get, response = api_response
    response.json.return_value = {"query": "open source", "chunks": [], "returned": 0}
    assert (
        ArcmiraSearchTool(api_key="explicit-key").run(query="open source")["returned"]
        == 0
    )
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer explicit-key"}


def test_missing_key_has_setup_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARCMIRA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Set ARCMIRA_API_KEY or pass api_key"):
        ArcmiraSearchTool()


@pytest.mark.parametrize(
    "arguments",
    [{"query": "x"}, {"query": "valid", "limit": 0}, {"query": "valid", "limit": 21}],
)
def test_invalid_input_is_rejected_before_request(
    arguments: dict[str, str | int], api_response: tuple[MagicMock, MagicMock]
) -> None:
    get, _ = api_response
    with pytest.raises(ValueError, match="arguments validation failed"):
        ArcmiraSearchTool(api_key="test-key").run(**arguments)
    get.assert_not_called()


@pytest.mark.parametrize(
    "status,code",
    [(401, "invalid_api_key"), (402, "quota_exceeded"), (429, "rate_limited")],
)
def test_api_errors_preserve_code_and_retry_information(
    status: int, code: str, api_response: tuple[MagicMock, MagicMock]
) -> None:
    get, response = api_response
    response.status_code = status
    response.json.return_value = {
        "error": {
            "code": code,
            "retry_after_seconds": 30,
            "doc_url": "https://arcmira.com/docs/errors",
        }
    }
    with pytest.raises(RuntimeError) as failure:
        ArcmiraSearchTool(api_key="test-key").run(query="open source")
    prefix = f"Arcmira search failed (HTTP {status}): "
    assert str(failure.value).startswith(prefix)
    assert json.loads(str(failure.value)[len(prefix) :]) == {
        "error": {
            "code": code,
            "retry_after_seconds": 30,
            "doc_url": "https://arcmira.com/docs/errors",
        }
    }
    assert get.call_count == 1


def test_network_failure_has_no_automatic_retry(
    api_response: tuple[MagicMock, MagicMock],
) -> None:
    get, _ = api_response
    get.side_effect = requests.Timeout("internal request context")
    with pytest.raises(RuntimeError) as failure:
        ArcmiraSearchTool(api_key="test-key").run(query="open source")
    assert (
        str(failure.value)
        == "Arcmira search could not complete the network request. No automatic retry was attempted."
    )
    assert get.call_count == 1


def test_non_json_error_omits_raw_response(
    api_response: tuple[MagicMock, MagicMock],
) -> None:
    _, response = api_response
    response.status_code = 502
    response.json.side_effect = ValueError("private proxy diagnostic")
    with pytest.raises(RuntimeError) as failure:
        ArcmiraSearchTool(api_key="test-key").run(query="open source")
    assert str(failure.value) == "Arcmira returned a non-JSON response (HTTP 502)."


def test_unexpected_json_shape_is_rejected(
    api_response: tuple[MagicMock, MagicMock],
) -> None:
    _, response = api_response
    response.json.return_value = ["not a search response"]
    with pytest.raises(RuntimeError, match="unexpected response format"):
        ArcmiraSearchTool(api_key="test-key").run(query="open source")


def test_catalog_exposes_the_tool_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crewai_tools.generate_tool_specs import ToolSpecExtractor

    monkeypatch.delenv("ARCMIRA_API_KEY", raising=False)
    specs = ToolSpecExtractor().extract_all_tools()
    entry = next(tool for tool in specs if tool["name"] == "ArcmiraSearchTool")
    assert entry["humanized_name"] == "Arcmira: YouTube Transcript Search"
    assert entry["env_vars"] == [
        {
            "name": "ARCMIRA_API_KEY",
            "description": "Arcmira API key, or pass api_key when creating the tool.",
            "required": True,
            "default": None,
        }
    ]
    assert set(entry["run_params_schema"]["properties"]) == {"query", "limit"}
    assert "api_key" not in entry["init_params_schema"]["properties"]
    stored = json.loads((Path(__file__).parents[2] / "tool.specs.json").read_text())
    assert (
        next(tool for tool in stored["tools"] if tool["name"] == "ArcmiraSearchTool")
        == entry
    )

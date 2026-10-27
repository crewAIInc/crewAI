"""Exercise the public tools and real SDK serialization without network access."""

from copy import deepcopy
import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError


pytest.importorskip("tavily")
from crewai_tools import TavilyExtractorTool, TavilySearchTool


SEARCH_DEFAULTS = {
    "query": "test query",
    "search_depth": "basic",
    "topic": "general",
    "days": 7,
    "max_results": 5,
    "include_answer": False,
    "include_raw_content": False,
    "include_images": False,
}
SEARCH_OPTIONS = {
    "chunks_per_source": 3,
    "include_domains_mode": "prefer",
    "include_domains": ["example.com"],
    "start_date": "2026-01-01",
    "end_date": "2026-10-01",
    "country": "united states",
    "language": "en",
    "filter_by_language": True,
    "auto_parameters": False,
    "exact_match": True,
    "safe_search": True,
    "include_image_descriptions": True,
    "include_images": True,
    "include_favicon": True,
    "include_usage": True,
    "include_published_date": True,
    "filter_by_published_date": True,
    "include_raw_content": "text",
    "max_results": 0,
}
EXTRACT_DEFAULTS = {
    "urls": ["https://example.com"],
    "extract_depth": "basic",
    "include_images": False,
    "timeout": 60,
}
EXTRACT_OPTIONS = {
    "query": "pricing",
    "chunks_per_source": 5,
    "format": "text",
    "include_favicon": True,
    "include_usage": True,
}
RESPONSE = {
    "results": [
        {
            "url": "https://example.com",
            "content": "a" * 1001,
            "raw_content": "b" * 1100,
            "favicon": "https://example.com/icon",
            "published_date": "2026-10-01",
            "score": 0.9,
        }
    ],
    "failed_results": [{"url": "https://example.org", "error": "unavailable"}],
    "usage": {"credits": 1},
    "request_id": "offline-request",
    "images": [],
}


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


def stub_transport(
    tool: TavilySearchTool | TavilyExtractorTool, monkeypatch: pytest.MonkeyPatch
) -> tuple[Mock, AsyncMock]:
    """Leave the SDK intact; intercept only its final HTTP send."""
    response = Mock(status_code=200)
    response.json.side_effect = lambda: deepcopy(RESPONSE)
    sync_post = Mock(return_value=response)
    async_post = AsyncMock(return_value=response)
    monkeypatch.setattr(tool.client.session, "post", sync_post)
    monkeypatch.setattr(tool.async_client._client, "post", async_post)
    return sync_post, async_post


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,arguments,defaults,options",
    [
        (TavilySearchTool, {"query": "test query"}, SEARCH_DEFAULTS, SEARCH_OPTIONS),
        (
            TavilyExtractorTool,
            {"urls": ["https://example.com"]},
            EXTRACT_DEFAULTS,
            EXTRACT_OPTIONS,
        ),
    ],
)
@pytest.mark.parametrize("configured", [False, True])
async def test_sync_async_requests_and_responses(
    monkeypatch: pytest.MonkeyPatch,
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    arguments: dict[str, Any],
    defaults: dict[str, Any],
    options: dict[str, Any],
    configured: bool,
) -> None:
    tool = tool_class(api_key="offline-test", **(options if configured else {}))
    sync_post, async_post = stub_transport(tool, monkeypatch)
    sync_result = json.loads(tool.run(**arguments))
    async_result = json.loads(await tool.arun(**arguments))
    expected_request = defaults | (options if configured else {})
    if tool_class is TavilySearchTool and configured:
        expected_request.pop("days")
    assert json.loads(sync_post.call_args.kwargs["data"]) == expected_request
    assert json.loads(async_post.call_args.kwargs["content"]) == expected_request
    assert sync_post.call_args.kwargs["timeout"] == 60
    assert async_post.call_args.kwargs["timeout"] == 60
    expected = deepcopy(RESPONSE)
    if tool_class is TavilySearchTool:
        expected["results"][0]["content"] = "a" * 1000 + "..."
    assert sync_result == async_result == expected
    assert tool.client.session.headers["X-Client-Name"] == "crewai"
    assert tool.async_client._client.headers["X-Client-Name"] == "crewai"
    await tool.async_client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "date_options",
    [
        {"start_date": "2026-01-01"},
        {"end_date": "2026-10-01"},
        {"start_date": "2026-01-01", "end_date": "2026-10-01"},
    ],
)
async def test_search_date_window_omits_implicit_days(
    monkeypatch: pytest.MonkeyPatch, date_options: dict[str, str]
) -> None:
    tool = TavilySearchTool(api_key="offline-test", **date_options)
    sync_post, async_post = stub_transport(tool, monkeypatch)

    def service_response(**kwargs: Any) -> Mock:
        body = json.loads(kwargs.get("data", kwargs.get("content")))
        if "days" in body:
            response = Mock(status_code=400)
            response.json.return_value = {
                "detail": {
                    "error": "When days is set, start_date or end_date cannot be set"
                }
            }
            return response
        response = Mock(status_code=200)
        response.json.return_value = deepcopy(RESPONSE)
        return response

    sync_post.side_effect = lambda *_args, **kwargs: service_response(**kwargs)
    async_post.side_effect = lambda *_args, **kwargs: service_response(**kwargs)
    try:
        assert json.loads(tool.run(query="test query"))["results"]
        assert json.loads(await tool.arun(query="test query"))["results"]
        for post, field in [(sync_post, "data"), (async_post, "content")]:
            body = json.loads(post.call_args.kwargs[field])
            assert "days" not in body
            assert date_options.items() <= body.items()
        assert tool.days == 7
    finally:
        await tool.async_client._client.aclose()


@pytest.mark.parametrize("days", [3, 7])
@pytest.mark.parametrize("date_field", ["start_date", "end_date"])
def test_search_rejects_explicit_days_with_date_bound(
    days: int, date_field: str
) -> None:
    with pytest.raises(ValidationError, match="days cannot be combined"):
        TavilySearchTool(
            api_key="offline-test", days=days, **{date_field: "2026-01-01"}
        )


@pytest.mark.asyncio
async def test_search_explicit_days_without_date_bounds_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = TavilySearchTool(api_key="offline-test", days=3, topic="news")
    sync_post, async_post = stub_transport(tool, monkeypatch)
    try:
        tool.run(query="test query")
        await tool.arun(query="test query")
        assert json.loads(sync_post.call_args.kwargs["data"])["days"] == 3
        assert json.loads(async_post.call_args.kwargs["content"])["days"] == 3
    finally:
        await tool.async_client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,arguments,options",
    [
        (
            TavilySearchTool,
            {"query": "test query"},
            {
                "auto_parameters": False,
                "exact_match": False,
                "safe_search": False,
                "filter_by_language": False,
                "include_image_descriptions": False,
                "include_favicon": False,
                "include_usage": False,
                "include_published_date": False,
                "filter_by_published_date": False,
                "max_results": 0,
                "max_content_length_per_result": 0,
            },
        ),
        (
            TavilyExtractorTool,
            {"urls": "https://example.com"},
            {
                "include_favicon": False,
                "include_usage": False,
            },
        ),
    ],
)
async def test_explicit_false_and_zero(
    monkeypatch: pytest.MonkeyPatch,
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    tool = tool_class(api_key="offline-test", **options)
    sync_post, async_post = stub_transport(tool, monkeypatch)
    sync_result = json.loads(tool.run(**arguments))
    async_result = json.loads(await tool.arun(**arguments))
    for post, keyword in [(sync_post, "data"), (async_post, "content")]:
        body = json.loads(post.call_args.kwargs[keyword])
        for key, value in options.items():
            if key != "max_content_length_per_result":
                assert body[key] == value
        assert "max_content_length_per_result" not in body
    assert sync_result == async_result
    if tool_class is TavilySearchTool:
        assert sync_result["results"][0]["content"] == "..."
    await tool.async_client._client.aclose()


@pytest.mark.parametrize("depth", ["basic", "advanced", "fast", "ultra-fast"])
@pytest.mark.parametrize("raw_content", [False, True, "markdown", "text"])
def test_search_depth_and_raw_content_modes(
    monkeypatch: pytest.MonkeyPatch, depth: str, raw_content: bool | str
) -> None:
    tool = TavilySearchTool(
        api_key="offline-test", search_depth=depth, include_raw_content=raw_content
    )
    sync_post, _ = stub_transport(tool, monkeypatch)
    tool.run(query="test query")
    body = json.loads(sync_post.call_args.kwargs["data"])
    assert body["search_depth"] == depth
    assert body["include_raw_content"] == raw_content


@pytest.mark.parametrize(
    "tool_class,options,message",
    [
        (TavilySearchTool, {"chunks_per_source": 0}, "greater than or equal to 1"),
        (TavilySearchTool, {"chunks_per_source": 4}, "less than or equal to 3"),
        (
            TavilySearchTool,
            {"search_depth": "ultra-fast", "chunks_per_source": 1},
            "ultra-fast",
        ),
        (
            TavilySearchTool,
            {"search_depth": "fast", "safe_search": True},
            "safe_search",
        ),
        (
            TavilySearchTool,
            {"search_depth": "ultra-fast", "safe_search": True},
            "safe_search",
        ),
        (TavilySearchTool, {"include_domains_mode": "prefer"}, "include_domains"),
        (TavilySearchTool, {"filter_by_language": True}, "language"),
        (TavilySearchTool, {"country": "united states", "topic": "news"}, "general"),
        (
            TavilyExtractorTool,
            {"chunks_per_source": 0, "query": "test"},
            "greater than or equal to 1",
        ),
        (
            TavilyExtractorTool,
            {"chunks_per_source": 6, "query": "test"},
            "less than or equal to 5",
        ),
        (TavilyExtractorTool, {"chunks_per_source": 1}, "requires query"),
        (TavilyExtractorTool, {"format": "html"}, "Input should be"),
    ],
)
def test_invalid_optional_configuration(
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    options: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        tool_class(api_key="offline-test", **options)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,arguments,options",
    [
        (TavilySearchTool, {"query": "test query"}, {"language": "en"}),
        (TavilyExtractorTool, {"urls": ["https://example.com"]}, {"format": "text"}),
    ],
)
async def test_agent_structured_tool_schema_and_execution(
    monkeypatch: pytest.MonkeyPatch,
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    arguments: dict[str, Any],
    options: dict[str, Any],
) -> None:
    tool = tool_class(api_key="offline-test", **options)
    sync_post, _ = stub_transport(tool, monkeypatch)
    structured = tool.to_structured_tool()
    schema = structured.args_schema.model_json_schema()
    assert set(schema["properties"]) == set(arguments)
    assert schema["required"] == list(arguments)
    expected = structured.invoke(json.dumps(arguments))
    assert await structured.ainvoke(json.dumps(arguments)) == expected
    body = json.loads(sync_post.call_args.kwargs["data"])
    for key, value in options.items():
        assert body[key] == value
    with pytest.raises(ValueError, match="validation failed"):
        structured.invoke({})
    await tool.async_client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,arguments",
    [
        (TavilySearchTool, {"query": "test query"}),
        (TavilyExtractorTool, {"urls": "https://example.com"}),
    ],
)
async def test_sdk_errors_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    arguments: dict[str, Any],
) -> None:
    tool = tool_class(api_key="offline-test")
    sync_post, async_post = stub_transport(tool, monkeypatch)
    failure = RuntimeError("offline simulated transport failure")
    sync_post.side_effect = failure
    async_post.side_effect = failure
    with pytest.raises(RuntimeError) as sync_error:
        tool.run(**arguments)
    with pytest.raises(RuntimeError) as async_error:
        await tool.arun(**arguments)
    assert sync_error.value is async_error.value is failure
    await tool.async_client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,arguments",
    [
        (TavilySearchTool, {"query": "test query"}),
        (TavilyExtractorTool, {"urls": "https://example.com"}),
    ],
)
async def test_fractional_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    arguments: dict[str, Any],
) -> None:
    tool = tool_class(api_key="offline-test", timeout=12.5)
    sync_post, async_post = stub_transport(tool, monkeypatch)
    tool.run(**arguments)
    await tool.arun(**arguments)
    assert sync_post.call_args.kwargs["timeout"] == 12.5
    assert async_post.call_args.kwargs["timeout"] == 12.5
    if tool_class is TavilyExtractorTool:
        assert json.loads(sync_post.call_args.kwargs["data"])["timeout"] == 12.5
        assert json.loads(async_post.call_args.kwargs["content"])["timeout"] == 12.5
    await tool.async_client._client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,arguments",
    [
        (TavilySearchTool, {"query": "test query"}),
        (TavilyExtractorTool, {"urls": "https://example.com"}),
    ],
)
async def test_structured_api_error_is_not_converted_to_success(
    monkeypatch: pytest.MonkeyPatch,
    tool_class: type[TavilySearchTool | TavilyExtractorTool],
    arguments: dict[str, Any],
) -> None:
    from tavily.errors import BadRequestError

    tool = tool_class(api_key="offline-test")
    sync_post, async_post = stub_transport(tool, monkeypatch)
    error_response = Mock(status_code=400)
    error_response.json.return_value = {"detail": {"error": "Invalid query"}}
    sync_post.return_value = error_response
    async_post.return_value = error_response
    with pytest.raises(BadRequestError, match="Invalid query"):
        tool.run(**arguments)
    with pytest.raises(BadRequestError, match="Invalid query"):
        await tool.arun(**arguments)
    await tool.async_client._client.aclose()

import json
import sys
from unittest.mock import patch

from crewai_tools import AdanosMarketSentimentTool
from crewai_tools.tools.adanos_tool.adanos_tool import OPERATIONS, AdanosToolInput
import httpx
import pytest


@pytest.fixture
def tool():
    return AdanosMarketSentimentTool(api_key="test-adanos-key")


@pytest.mark.parametrize(
    "source,operation",
    [
        (source, operation)
        for source, operations in OPERATIONS.items()
        for operation in sorted(operations)
    ],
)
def test_every_operation_uses_official_sdk_route(tool, source, operation):
    params = {}
    prefix = f"/{source}/stocks/v1"
    if source == "crypto":
        prefix = "/reddit/crypto/v1"
    asset = "token/BTC" if source == "crypto" else "stock/AAPL"
    suffix = {
        "stock": asset,
        "token": asset,
        "mentions": f"{asset}/mentions",
        "explain": f"{asset}/explain",
        "trending_sectors": "trending/sectors",
        "trending_countries": "trending/countries",
        "market_sentiment": "market-sentiment",
    }.get(operation, operation)
    path = f"{prefix}/{suffix}"
    if operation in {"stock", "token", "mentions", "explain"}:
        params = {"symbol": "BTC"} if source == "crypto" else {"ticker": "AAPL"}
    if operation == "compare":
        params = (
            {"symbols": ["BTC", "ETH"]}
            if source == "crypto"
            else {"tickers": ["AAPL", "MSFT"]}
        )
    if operation == "search":
        params = {"query": "Apple"}
    if source == "sentiment":
        params, path = {"text": "Earnings improved."}, "/sentiment/v1/analyze"
    if source == "status":
        path = "/health"

    def send(client, request, **kwargs):
        assert request.url.host == "api.adanos.org"
        assert request.url.path == path
        assert request.method == ("POST" if operation == "analyze" else "GET")
        assert request.headers["X-API-Key"] == "test-adanos-key"
        assert "days" not in request.url.params
        return httpx.Response(
            401, json={"detail": "private server detail"}, request=request
        )

    with patch.object(httpx.Client, "send", autospec=True, side_effect=send) as request:
        result = tool.run(source=source, operation=operation, parameters=params)
    assert request.call_count == 1
    assert "error" in result
    assert "data" not in result
    assert "private server detail" not in json.dumps(result)


def test_stock_preserves_source_values_and_dates(tool):
    payload = {
        "ticker": "AAPL",
        "found": True,
        "buzz_score": 12.5,
        "sentiment_score": -0.2,
        "mentions": 0,
        "future_metric": None,
    }

    def send(client, request, **kwargs):
        assert dict(request.url.params) == {"from": "2026-09-20", "to": "2026-09-21"}
        return httpx.Response(200, json=payload, request=request)

    with patch.object(httpx.Client, "send", autospec=True, side_effect=send):
        result = tool.run(
            source="reddit",
            operation="stock",
            parameters={
                "ticker": "AAPL",
                "from_": "2026-09-20",
                "to": "2026-09-21",
            },
        )
    assert result == {"source": "reddit", "operation": "stock", "data": payload}


@pytest.mark.parametrize(
    "status,body",
    [
        (403, {"detail": {"error": "historical_limit", "message": "private"}}),
        (404, {"detail": {"error": "unsupported_asset", "message": "private"}}),
        (
            422,
            {
                "detail": [
                    {"loc": ["query", "limit"], "msg": "private", "type": "value_error"}
                ]
            },
        ),
        (429, {"detail": "quota exhausted: test-adanos-key"}),
        (500, {"detail": "private"}),
        (200, {"malformed": True}),
    ],
)
def test_errors_are_not_sentiment_or_leaked(tool, status, body):
    with patch.object(
        httpx.Client,
        "send",
        autospec=True,
        side_effect=lambda client, request, **kwargs: httpx.Response(
            status, json=body, request=request
        ),
    ):
        result = tool.run(
            source="reddit", operation="stock", parameters={"ticker": "AAPL"}
        )
    assert "error" in result and "data" not in result
    assert "private" not in json.dumps(result)
    assert "test-adanos-key" not in json.dumps(result)


@pytest.mark.parametrize(
    "source,operation,parameters",
    [
        ("crypto", "stock", {"ticker": "BTC"}),
        ("polymarket", "explain", {"ticker": "AAPL"}),
        ("reddit", "__getattribute__", {}),
        ("reddit", "trending", {"days": 7}),
        ("reddit", "trending", {"from_": "2026-09-20"}),
        ("reddit", "trending", {"from_": "2026-09-21", "to": "2026-09-20"}),
        ("reddit", "trending", {"from_": "20260920", "to": "20260921"}),
        ("reddit", "trending", {"base_url": "https://example.org"}),
        ("reddit", "trending", {"api_key": "untrusted"}),
        ("reddit", "stock", {}),
        ("status", "health", {"headers": {}}),
    ],
)
def test_invalid_inputs_do_not_make_requests(tool, source, operation, parameters):
    with patch.object(httpx.Client, "send") as request:
        result = tool._run(source, operation, parameters)
    assert result["error"] == "invalid_arguments"
    request.assert_not_called()


def test_configuration_and_agent_schema(monkeypatch):
    monkeypatch.setenv("ADANOS_API_KEY", "test-adanos-key")
    tool = AdanosMarketSentimentTool()
    assert "test-adanos-key" not in repr(tool)
    assert "api_key" not in tool.model_dump()
    assert set(AdanosToolInput.model_json_schema()["properties"]) == {
        "source",
        "operation",
        "parameters",
    }
    monkeypatch.delenv("ADANOS_API_KEY")
    with patch.object(httpx.Client, "send") as request:
        assert (
            AdanosMarketSentimentTool().run(source="reddit", operation="trending")[
                "error"
            ]
            == "configuration_error"
        )
    request.assert_not_called()


def test_optional_sdk_is_only_needed_on_use(tool):
    with patch.dict(sys.modules, {"adanos": None}):
        with pytest.raises(ImportError, match="crewai-tools"):
            tool.run(source="reddit", operation="trending")


def test_timeout_closes_client_without_retry(tool):
    with (
        patch.object(
            httpx.Client, "send", side_effect=httpx.ReadTimeout("private")
        ) as send,
        patch.object(httpx.Client, "__exit__") as close,
    ):
        result = tool.run(source="reddit", operation="trending")
    assert result["error"] == "request_failed"
    assert send.call_count == 1
    close.assert_called_once()


@pytest.mark.parametrize(
    "source,operation,parameters,payload",
    [
        (
            "sentiment",
            "analyze",
            {"text": "Improving margins."},
            {"sentiment_score": 0.2},
        ),
        (
            "reddit",
            "search",
            {"query": "Apple"},
            {"query": "Apple", "results": [], "count": 0, "period_days": 1},
        ),
    ],
)
def test_dict_and_list_results(tool, source, operation, parameters, payload):
    with patch.object(
        httpx.Client,
        "send",
        autospec=True,
        side_effect=lambda client, request, **kwargs: httpx.Response(
            200, json=payload, request=request
        ),
    ):
        result = tool.run(source=source, operation=operation, parameters=parameters)
    assert result["data"] == payload


@pytest.mark.parametrize(
    "operation,suffix",
    [
        ("trending", "trending"),
        ("trending_sectors", "trending/sectors"),
        ("trending_countries", "trending/countries"),
    ],
)
def test_news_publisher_filter(tool, operation, suffix):
    def send(client, request, **kwargs):
        assert request.url.path == f"/news/stocks/v1/{suffix}"
        assert request.url.params["source"] == "reuters"
        return httpx.Response(401, json={"detail": "Unauthorized"}, request=request)

    with patch.object(httpx.Client, "send", autospec=True, side_effect=send) as request:
        result = tool.run(
            source="news", operation=operation, parameters={"source": "reuters"}
        )
    assert request.call_count == 1
    assert result["error"] == "api_error"

import json
from typing import Any

from crewai.tools.tool_failure import ToolFailure
from crewai_tools import (
    DecodoEcommerceTool,
    DecodoSearchTool,
    DecodoSERPTool,
    DecodoWebScraperTool,
)
from crewai_tools.tools.decodo_ecommerce_tool.decodo_ecommerce_tool import (
    DecodoEcommerceArgs,
)
import httpx
import pytest


class FakeApi:
    """Stands in for httpx.request inside the SDK and records what was sent."""

    def __init__(self, status_code: int = 200, body: Any = None) -> None:
        self.status_code = status_code
        self.body = body if body is not None else {"results": [{"content": "# Hello", "status_code": 200}]}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, method: str, url: str, *, headers: dict[str, str], json: Any, timeout: float) -> httpx.Response:
        self.calls.append({"method": method, "url": url, "headers": headers, "json": json})
        return httpx.Response(self.status_code, json=self.body)

    @property
    def payload(self) -> dict[str, Any]:
        return self.calls[-1]["json"]  # type: ignore[no-any-return]


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    api = FakeApi()
    monkeypatch.setattr("decodo.http.httpx.request", api)
    return api


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECODO_API_KEY", "test-key")


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DECODO_API_KEY")
    with pytest.raises(ValueError, match="DECODO_API_KEY"):
        DecodoWebScraperTool()


def test_blank_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DECODO_API_KEY", " \n")
    with pytest.raises(ValueError, match="DECODO_API_KEY"):
        DecodoWebScraperTool()


@pytest.mark.parametrize("api_key", ["padded-key\n", " padded-key "])
def test_api_key_whitespace_is_stripped(fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    monkeypatch.setenv("DECODO_API_KEY", api_key)
    DecodoWebScraperTool().run(url="https://example.com")
    DecodoWebScraperTool(api_key=api_key).run(url="https://example.com")
    assert [call["headers"]["Authorization"] for call in fake_api.calls] == ["Bearer padded-key"] * 2


def test_api_key_argument_overrides_env(fake_api: FakeApi) -> None:
    DecodoWebScraperTool(api_key="explicit-key").run(url="https://example.com")
    assert fake_api.calls[0]["headers"]["Authorization"] == "Bearer explicit-key"


def test_requests_go_to_data_api_with_crewai_integration_header(fake_api: FakeApi) -> None:
    DecodoWebScraperTool().run(url="https://example.com")

    call = fake_api.calls[0]
    assert call["url"].startswith("https://data.decodo.com/")
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["headers"]["x-integration"] == "crewai-python"


def test_web_scraper_returns_markdown(fake_api: FakeApi) -> None:
    result = DecodoWebScraperTool().run(url="https://example.com")

    assert result == "# Hello"
    assert fake_api.payload == {"target": "universal", "url": "https://example.com", "markdown": True}


def test_web_scraper_js_render_and_geo(fake_api: FakeApi) -> None:
    DecodoWebScraperTool().run(url="https://example.com", js_render=True, geo="Germany")

    assert fake_api.payload["headless"] == "html"
    assert fake_api.payload["geo"] == "Germany"


def test_web_scraper_truncates_long_pages(fake_api: FakeApi) -> None:
    fake_api.body = {"results": [{"content": "a" * 50, "status_code": 200}]}

    result = DecodoWebScraperTool(max_chars=10).run(url="https://example.com")

    assert result.startswith("a" * 10 + "\n\n[Truncated to 10 of 50 characters.]")


def test_search_google_returns_full_parsed_results(fake_api: FakeApi) -> None:
    content = {"results": {"organic": [{"title": "A", "images": ["x"]}], "images": ["y"]}}
    fake_api.body = {"results": [{"status_code": 200, "content": content}]}

    result = DecodoSearchTool().run(query="coffee")

    assert fake_api.payload == {"target": "google_search", "query": "coffee", "parse": True}
    assert json.loads(result) == content


def test_search_amazon(fake_api: FakeApi) -> None:
    content = {"results": {"organic": [], "suggested": [1]}}
    fake_api.body = {"results": [{"status_code": 200, "content": content}]}

    result = DecodoSearchTool().run(query="keyboard", source="amazon", geo="United States")

    assert fake_api.payload == {"target": "amazon_search", "query": "keyboard", "geo": "United States", "parse": True}
    assert json.loads(result) == content


@pytest.mark.parametrize(
    ("status_code", "code", "retryable"),
    [
        (401, "authentication_failed", False),
        (422, "invalid_params", False),
        (429, "rate_limited", True),
        (500, "500", True),
    ],
)
def test_api_errors_become_tool_failures(fake_api: FakeApi, status_code: int, code: str, retryable: bool) -> None:
    fake_api.status_code = status_code
    fake_api.body = {"message": "nope"}

    result = DecodoWebScraperTool()._run(url="https://example.com")

    assert isinstance(result, ToolFailure)
    assert result.code == code
    assert result.retryable is retryable


def test_network_errors_become_retryable_tool_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr("decodo.http.httpx.request", refuse)

    for tool, kwargs in [
        (DecodoWebScraperTool(), {"url": "https://example.com"}),
        (DecodoSearchTool(), {"query": "coffee"}),
        (DecodoEcommerceTool(), {"store": "amazon", "query": "coffee"}),
        (DecodoSERPTool(), {"query": "coffee"}),
    ]:
        result = tool.run(**kwargs)
        assert isinstance(result, ToolFailure)
        assert result.code == "network_error"
        assert result.retryable is True
        assert "Connection refused" in result.message


@pytest.mark.parametrize(
    "error",
    [
        httpx.LocalProtocolError("Illegal header value b'Bearer secret\\nkey'"),
        UnicodeEncodeError("ascii", "Bearer secretkeyé", 16, 17, "ordinal not in range(128)"),
    ],
)
def test_unsendable_api_key_is_not_echoed(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def reject_header(*args: Any, **kwargs: Any) -> httpx.Response:
        raise error

    monkeypatch.setattr("decodo.http.httpx.request", reject_header)

    result = DecodoWebScraperTool().run(url="https://example.com")

    assert isinstance(result, ToolFailure)
    assert result.code == "invalid_request"
    assert "secret" not in result.message


def test_target_error_status_becomes_tool_failure(fake_api: FakeApi) -> None:
    fake_api.body = {"results": [{"content": "", "status_code": 404}]}

    result = DecodoWebScraperTool()._run(url="https://example.com/missing")

    assert isinstance(result, ToolFailure)
    assert result.code == "404"


def test_empty_results_become_tool_failure(fake_api: FakeApi) -> None:
    fake_api.body = {"results": []}

    result = DecodoSearchTool()._run(query="coffee")

    assert isinstance(result, ToolFailure)
    assert result.code == "empty_response"


def test_failed_task_becomes_retryable_tool_failure(fake_api: FakeApi) -> None:
    fake_api.body = {"status": "failed", "status_code": 613, "message": "We were not able to scrape the target."}

    result = DecodoWebScraperTool()._run(url="https://example.com")

    assert isinstance(result, ToolFailure)
    assert result.code == "613"
    assert result.retryable is True
    assert "We were not able to scrape the target." in result.message


@pytest.mark.parametrize(
    ("kwargs", "payload"),
    [
        (
            {"store": "amazon", "product_id": "B09H74FXNW", "zip_code": "10001"},
            {"target": "amazon_product", "query": "B09H74FXNW", "geo": "10001", "parse": True},
        ),
        (
            {"store": "amazon", "query": "usb c cable"},
            {"target": "amazon_search", "query": "usb c cable", "parse": True},
        ),
        (
            {"store": "walmart", "product_id": "15296401808"},
            {"target": "walmart_product", "product_id": "15296401808", "parse": True},
        ),
        (
            {"store": "walmart", "query": "camping tent", "zip_code": "10001"},
            {"target": "walmart_search", "query": "camping tent", "delivery_zip": "10001", "parse": True},
        ),
        (
            {"store": "target", "product_id": "1003921355"},
            {"target": "target_product", "product_id": "1003921355", "headless": "html", "parse": True},
        ),
        (
            {"store": "target", "query": "camping tent"},
            {"target": "target_search", "query": "camping tent", "headless": "html", "parse": True},
        ),
    ],
)
def test_ecommerce_requests(fake_api: FakeApi, kwargs: dict[str, Any], payload: dict[str, Any]) -> None:
    content = {"title": "A product", "price": 9.99}
    fake_api.body = {"results": [{"status_code": 200, "content": content}]}

    result = DecodoEcommerceTool().run(**kwargs)

    assert fake_api.payload == payload
    assert json.loads(result) == content


@pytest.mark.parametrize("kwargs", [{}, {"product_id": "B09H74FXNW", "query": "usb c cable"}])
def test_ecommerce_needs_exactly_one_of_product_id_or_query(fake_api: FakeApi, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="either product_id or query"):
        DecodoEcommerceArgs(store="amazon", **kwargs)

    result = DecodoEcommerceTool()._run(store="amazon", **kwargs)

    assert isinstance(result, ToolFailure)
    assert result.code == "invalid_params"
    assert not fake_api.calls


@pytest.mark.parametrize(
    ("kwargs", "payload"),
    [
        ({"query": "batteries"}, {"target": "google_search", "query": "batteries", "parse": True}),
        (
            {"query": "batteries", "page": 2, "geo": "Germany"},
            {"target": "google_search", "query": "batteries", "geo": "Germany", "page_from": 2, "parse": True},
        ),
        (
            {"query": "batteries", "engine": "bing", "page": 3},
            {"target": "bing_search", "query": "batteries", "page_from": 3, "headless": "html", "parse": True},
        ),
        (
            {"query": "what are batteries", "engine": "google_ai_mode", "page": 2},
            {"target": "google_ai_mode", "query": "what are batteries", "parse": True},
        ),
    ],
)
def test_serp_requests(fake_api: FakeApi, kwargs: dict[str, Any], payload: dict[str, Any]) -> None:
    content = {"results": {"organic": [{"title": "A"}]}}
    fake_api.body = {"results": [{"status_code": 200, "content": content}]}

    result = DecodoSERPTool().run(**kwargs)

    assert fake_api.payload == payload
    assert json.loads(result) == content

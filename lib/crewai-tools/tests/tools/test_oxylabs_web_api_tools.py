import json
from typing import Any
from unittest.mock import MagicMock

from crewai.tools.tool_failure import ToolFailure, ToolFailureReason
from crewai_tools import OxylabsWebScrapeTool, OxylabsWebSearchTool
from crewai_tools.tools.oxylabs_web_api_base_tool import oxylabs_web_api_base_tool
from crewai_tools.tools.oxylabs_web_scrape_tool import oxylabs_web_scrape_tool
import pytest
import requests


SEARCH_BODY = {
    "state": "done",
    "results": [
        {
            "title": "Etno Dvaras",
            "outline": ["Traditional Lithuanian food."],
            "url": "https://example.com/etno",
            "metadata": {"position": 1},
        }
    ],
    "related_questions": [
        {"question": "Where to eat in Vilnius?", "title": None, "snippet": None}
    ],
    "related_searches": [{"query": "vilnius restaurants"}],
}

PAGE = "# Page\n" + "text " * 200


def scrape_body(key: str = "markdown", value: Any = PAGE) -> dict[str, Any]:
    return {"state": "done", "results": [{key: value, "metadata": {"page": 1}}]}


def reply(status: int, body: Any) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = body
    response.text = json.dumps(body)
    return response


class FakeWebAPI:
    """Answers requests from a queue of replies, in order; the last one repeats."""

    def __init__(self) -> None:
        self.replies: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def add(self, status: int, body: Any) -> "FakeWebAPI":
        self.replies.append(reply(status, body))
        return self

    def fail_with(self, error: Exception) -> "FakeWebAPI":
        self.replies.append(error)
        return self

    def __call__(self, method: str, url: str, **kwargs: Any) -> Any:
        self.calls.append({"method": method, "url": url, **kwargs})
        answer = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    @property
    def paths(self) -> list[str]:
        return [
            f"{c['method']} {c['url'].removeprefix('https://webapi.oxylabs.io')}"
            for c in self.calls
        ]

    @property
    def last_payload(self) -> dict[str, Any]:
        return self.calls[-1]["json"]


@pytest.fixture
def web_api(monkeypatch: pytest.MonkeyPatch) -> FakeWebAPI:
    fake = FakeWebAPI()
    monkeypatch.setattr(oxylabs_web_api_base_tool.requests, "request", fake)
    # Retries and polling would otherwise sleep for real.
    monkeypatch.setattr(oxylabs_web_api_base_tool.time, "sleep", lambda _: None)
    monkeypatch.setattr(oxylabs_web_scrape_tool.time, "sleep", lambda _: None)
    return fake


def call(tool: Any, web_api: FakeWebAPI) -> None:
    """Make one successful call with whichever Web API tool this is."""
    if isinstance(tool, OxylabsWebSearchTool):
        web_api.add(200, SEARCH_BODY)
        tool.run(query="q")
    else:
        web_api.add(200, scrape_body())
        tool.run(url="https://example.com")


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OXYLABS_WEB_API_KEY", "test-key")


@pytest.mark.parametrize("tool_class", [OxylabsWebSearchTool, OxylabsWebScrapeTool])
class TestInitialization:
    def test_reads_the_key_from_the_environment(
        self, tool_class: type, web_api: FakeWebAPI
    ) -> None:
        call(tool_class(), web_api)
        assert web_api.calls[-1]["headers"]["Authorization"] == "Bearer test-key"

    def test_explicit_key_wins(self, tool_class: type, web_api: FakeWebAPI) -> None:
        call(tool_class(api_key="other-key"), web_api)
        assert web_api.calls[-1]["headers"]["Authorization"] == "Bearer other-key"

    def test_missing_key_raises(
        self, tool_class: type, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OXYLABS_WEB_API_KEY")
        with pytest.raises(ValueError, match="OXYLABS_WEB_API_KEY"):
            tool_class()

    def test_key_is_not_serialized(self, tool_class: type) -> None:
        tool = tool_class(api_key="secret-key")
        assert "secret-key" not in repr(tool)
        assert "secret-key" not in tool.model_dump_json()


class TestWebSearch:
    def test_returns_compact_results(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, SEARCH_BODY)
        out = json.loads(OxylabsWebSearchTool().run(query="restaurants"))
        assert out == {
            "results": [
                {
                    "title": "Etno Dvaras",
                    "url": "https://example.com/etno",
                    "outline": ["Traditional Lithuanian food."],
                }
            ],
            "related_questions": ["Where to eat in Vilnius?"],
        }

    def test_sends_query_options(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, SEARCH_BODY)
        OxylabsWebSearchTool().run(query="q", max_results=3, location="de")
        assert web_api.paths == ["POST /v1/search"]
        assert web_api.last_payload == {"query": "q", "max_results": 3, "location": "DE"}

    def test_omitted_location_is_not_sent(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, SEARCH_BODY)
        OxylabsWebSearchTool().run(query="q")
        assert "location" not in web_api.last_payload

    def test_identifies_crewai(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, SEARCH_BODY)
        OxylabsWebSearchTool().run(query="q")
        sdk = web_api.calls[-1]["headers"]["x-oxylabs-sdk"]
        assert sdk.startswith("oxylabs-crewai-sdk-python/")

    def test_blank_query_is_rejected_without_a_call(self, web_api: FakeWebAPI) -> None:
        result = OxylabsWebSearchTool()._run(query="   ")
        assert isinstance(result, ToolFailure)
        assert result.reason is ToolFailureReason.INVALID_INPUT
        assert web_api.calls == []


class TestWebScrape:
    def test_returns_markdown(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body())
        assert OxylabsWebScrapeTool().run(url="https://example.com") == PAGE
        assert web_api.last_payload == {"url": "https://example.com", "output": ["markdown"]}

    def test_sends_page_options(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body("html", "<html>" + "x" * 600 + "</html>"))
        OxylabsWebScrapeTool().run(
            url="https://example.com",
            output="html",
            location="us",
            device="mobile",
            run_js=True,
        )
        assert web_api.last_payload == {
            "url": "https://example.com",
            "output": ["html"],
            "location": "US",
            "device": "mobile",
            "run_js": True,
        }

    def test_json_prompt_returns_the_fields(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body("json", {"price": "91,99 €"}))
        out = OxylabsWebScrapeTool().run(url="https://example.com", json_prompt="the price")
        assert json.loads(out) == {"price": "91,99 €"}
        assert web_api.last_payload["output"] == ["json"]
        assert web_api.last_payload["json"] == {"prompt": "the price"}

    def test_long_pages_are_truncated(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body(value="a" * 1000))
        out = OxylabsWebScrapeTool(max_content_chars=100).run(url="https://example.com")
        assert out.startswith("a" * 100 + "\n\n[Truncated: showing 100 of 1000")

    def test_truncation_can_be_turned_off(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body(value="a" * 60_000))
        out = OxylabsWebScrapeTool(max_content_chars=None).run(url="https://example.com")
        assert out == "a" * 60_000

    def test_near_empty_page_suggests_run_js(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body(value="Loading..."))
        assert "run_js" in OxylabsWebScrapeTool().run(url="https://example.com")

    def test_no_run_js_hint_once_rendered(self, web_api: FakeWebAPI) -> None:
        web_api.add(200, scrape_body(value="Loading..."))
        out = OxylabsWebScrapeTool().run(url="https://example.com", run_js=True)
        assert out == "Loading..."

    def test_relative_url_is_rejected_without_a_call(self, web_api: FakeWebAPI) -> None:
        result = OxylabsWebScrapeTool()._run(url="example.com")
        assert isinstance(result, ToolFailure)
        assert result.reason is ToolFailureReason.INVALID_INPUT
        assert web_api.calls == []

    def test_timed_out_scrape_is_queued_and_collected(self, web_api: FakeWebAPI) -> None:
        web_api.add(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.add(202, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.add(200, {"state": "pending", "metadata": {"request_id": "7"}})
        web_api.add(200, scrape_body())
        out = OxylabsWebScrapeTool().run(url="https://example.com", run_js=True)
        assert out == PAGE
        assert web_api.paths == [
            "POST /v1/scrape",
            "POST /v1/async/scrape",
            "GET /v1/async/scrape/7",
            "GET /v1/async/scrape/7",
        ]

    def test_gives_up_on_a_queued_scrape_after_max_wait(
        self, web_api: FakeWebAPI
    ) -> None:
        web_api.add(504, {"title": "REQUEST_FAILED_SCRAPE_TIMEOUT"})
        web_api.add(200, {"state": "pending", "metadata": {"request_id": "7"}})
        result = OxylabsWebScrapeTool(max_wait=0)._run(url="https://example.com")
        assert isinstance(result, ToolFailure)
        assert result.code == "pending"
        assert result.retryable
        assert result.details == {"request_id": "7"}


class TestFailures:
    """Failures come back as a ToolFailure the agent can act on, never a raise."""

    def run_search(self) -> Any:
        return OxylabsWebSearchTool()._run(query="q")

    def test_bad_key(self, web_api: FakeWebAPI) -> None:
        web_api.add(401, {})
        result = self.run_search()
        assert isinstance(result, ToolFailure)
        assert result.code == "401"
        assert "OXYLABS_WEB_API_KEY" in result.message
        assert not result.retryable

    def test_spent_quota_is_not_retried(self, web_api: FakeWebAPI) -> None:
        web_api.add(429, {"title": "QUOTA_EXCEEDED"})
        result = self.run_search()
        assert isinstance(result, ToolFailure)
        assert result.code == "quota_exceeded"
        assert len(web_api.calls) == 1

    def test_rate_limit_is_retried_then_reported(self, web_api: FakeWebAPI) -> None:
        web_api.add(429, {"title": "TOO_MANY_REQUESTS"})
        result = OxylabsWebSearchTool(max_retries=1)._run(query="q")
        assert isinstance(result, ToolFailure)
        assert result.code == "429"
        assert result.retryable
        assert len(web_api.calls) == 2

    def test_transient_error_is_retried(self, web_api: FakeWebAPI) -> None:
        web_api.add(503, {"detail": "busy"}).add(200, SEARCH_BODY)
        assert "Etno Dvaras" in OxylabsWebSearchTool().run(query="q")
        assert len(web_api.calls) == 2

    def test_connection_error_is_retried(self, web_api: FakeWebAPI) -> None:
        web_api.fail_with(requests.ConnectionError("down")).add(200, SEARCH_BODY)
        assert "Etno Dvaras" in OxylabsWebSearchTool().run(query="q")

    def test_connection_error_after_retries(self, web_api: FakeWebAPI) -> None:
        web_api.fail_with(requests.ConnectionError("down"))
        result = self.run_search()
        assert isinstance(result, ToolFailure)
        assert result.code == "connection_error"
        assert result.retryable

    def test_faulted_request_is_not_resent(self, web_api: FakeWebAPI) -> None:
        web_api.add(
            500,
            {
                "state": "faulted",
                "title": "REQUEST_FAILED_AFTER_TOO_MANY_RETRIES",
                "detail": "Try again.",
                "metadata": {"request_id": "99"},
            },
        )
        result = OxylabsWebScrapeTool()._run(url="https://example.com")
        assert isinstance(result, ToolFailure)
        assert result.code == "REQUEST_FAILED_AFTER_TOO_MANY_RETRIES"
        assert result.details == {"request_id": "99"}
        assert len(web_api.calls) == 1

    def test_validation_error_names_the_field(self, web_api: FakeWebAPI) -> None:
        web_api.add(
            400,
            {
                "title": "VALIDATION_ERROR",
                "errors": [{"pointer": "#/location", "detail": "bad pattern"}],
            },
        )
        result = self.run_search()
        assert isinstance(result, ToolFailure)
        assert "#/location: bad pattern" in result.message
        assert not result.retryable

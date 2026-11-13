from typing import Any

import pytest
import requests

from crewai_tools import (
    AlionCompanyProfileTool,
    AlionCompanySearchTool,
    AlionJobSearchTool,
    AlionSimilarCompaniesTool,
)


class FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self) -> Any:
        return self.payload


class Recorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.queue: list[FakeResponse] = []

    def post(self, url: str, json: Any, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "body": json, "headers": headers})
        return self.queue.pop(0)


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    monkeypatch.delenv("ALION_API_KEY", raising=False)
    recorder = Recorder()
    monkeypatch.setattr(requests, "post", recorder.post)
    return recorder


def ok(structured: Any) -> FakeResponse:
    return FakeResponse({"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": structured, "content": []}})


def test_company_search_without_key_uses_the_open_address(http: Recorder) -> None:
    http.queue.append(ok({"items": [{"name": "Stripe", "domain": "stripe.com"}]}))

    result = AlionCompanySearchTool()._run(industry="Fintech", country="DE", startup=True, limit=5)

    assert result == {"success": True, "result": {"items": [{"name": "Stripe", "domain": "stripe.com"}]}}
    assert http.calls[0]["url"] == "https://alion.io/mcp/open"
    assert "Authorization" not in http.calls[0]["headers"]
    assert http.calls[0]["body"]["params"] == {
        "name": "search_companies",
        "arguments": {"industry": "Fintech", "country": "DE", "startup": True, "limit": 5, "detail": "compact"},
    }


def test_key_from_the_environment_uses_the_full_server(http: Recorder, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALION_API_KEY", "test-key")
    http.queue.append(ok({"name": "Stripe", "funding": {"last_round": "series_g"}}))

    result = AlionCompanyProfileTool()._run(domain="stripe.com")

    assert result["success"] is True
    assert http.calls[0]["url"] == "https://alion.io/mcp"
    assert http.calls[0]["headers"]["Authorization"] == "Bearer test-key"
    assert http.calls[0]["body"]["params"]["name"] == "get_company"


def test_profile_without_key_falls_back_to_identity(http: Recorder) -> None:
    http.queue.append(ok({"match": {"name": "Stripe", "domain": "stripe.com"}}))

    result = AlionCompanyProfileTool()._run(domain="stripe.com")

    assert http.calls[0]["body"]["params"]["name"] == "resolve_company"
    assert result["success"] is True and "ALION_API_KEY" in result["note"]


def test_profile_needs_a_domain_or_name(http: Recorder) -> None:
    assert AlionCompanyProfileTool()._run() == {"success": False, "error": "Give a company domain or name"}
    assert http.calls == []


def test_tool_refusal_is_reported(http: Recorder) -> None:
    refusal = {"error": "rate_limited", "message": "Too many calls in a minute"}
    http.queue.append(FakeResponse({"jsonrpc": "2.0", "id": 1, "result": {"structuredContent": refusal, "isError": True}}))

    assert AlionJobSearchTool()._run(role="Data Engineer") == {"success": False, "error": "Too many calls in a minute"}


def test_http_error_is_reported(http: Recorder) -> None:
    http.queue.append(FakeResponse({}, status=502))

    result = AlionSimilarCompaniesTool()._run(company="stripe.com")

    assert result["success"] is False and "502" in result["error"]


def test_none_arguments_are_dropped(http: Recorder) -> None:
    http.queue.append(ok({"items": []}))

    AlionJobSearchTool()._run(role="Backend", country=None, limit=3)

    assert http.calls[0]["body"]["params"]["arguments"] == {"role": "Backend", "limit": 3, "detail": "compact"}


def test_error_text_without_structured_content_reaches_the_agent(http: Recorder) -> None:
    http.queue.append(
        FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "content": [{"type": "text", "text": "Rate limit: 30 calls a minute"}],
                    "isError": True,
                },
            }
        )
    )

    assert AlionJobSearchTool()._run(role="Backend") == {
        "success": False,
        "error": "Rate limit: 30 calls a minute",
    }


def test_json_error_text_without_structured_content_is_parsed(http: Recorder) -> None:
    text = '{"error": "invalid_params", "message": "limit must be 1-50"}'
    http.queue.append(
        FakeResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": text}], "isError": True},
            }
        )
    )

    assert AlionCompanySearchTool()._run(limit=5) == {
        "success": False,
        "error": "limit must be 1-50",
    }

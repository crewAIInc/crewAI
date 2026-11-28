import json
from unittest.mock import patch

import httpx
import pytest

from crewai_tools.tools.alterlab_tool import alterlab_tool as module
from crewai_tools.tools.alterlab_tool.alterlab_tool import (
    AlterLabCrawlTool,
    AlterLabScrapeTool,
    AlterLabSearchTool,
)


class Recorder:
    """Serve canned responses through an httpx MockTransport and record requests."""

    def __init__(self, *responses: httpx.Response):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


@pytest.fixture
def serve():
    def _serve(*responses: httpx.Response) -> Recorder:
        recorder = Recorder(*responses)
        real_client = httpx.Client

        def make_client(**kwargs):
            return real_client(transport=httpx.MockTransport(recorder.handler), **kwargs)

        patcher = patch.object(module.httpx, "Client", make_client)
        patcher.start()
        serve.patchers.append(patcher)
        return recorder

    serve.patchers = []
    yield _serve
    for patcher in serve.patchers:
        patcher.stop()


@pytest.fixture(autouse=True)
def no_sleep():
    with patch.object(module.time, "sleep"):
        yield


def ok(data, status=200):
    return httpx.Response(status, json=data)


def test_scrape_returns_markdown_and_sends_auth(serve):
    rec = serve(ok({"content": {"markdown": "# Hello", "text": "Hello"}}))

    result = AlterLabScrapeTool(api_key="sk_test").run(url="https://example.com")

    assert result == "# Hello"
    request = rec.requests[0]
    assert request.url == "https://api.alterlab.io/api/v1/scrape"
    assert request.headers["X-API-Key"] == "sk_test"
    body = json.loads(request.content)
    assert body["url"] == "https://example.com"
    assert body["sync"] is True
    assert body["formats"] == ["markdown"]


def test_scrape_falls_back_to_text_and_handles_empty(serve):
    serve(ok({"content": {"text": "plain"}}), ok({"content": {}}))
    tool = AlterLabScrapeTool(api_key="sk_test")

    assert tool.run(url="https://example.com") == "plain"
    assert "No content" in tool.run(url="https://example.com")


def test_scrape_params_are_forwarded(serve):
    rec = serve(ok({"content": "body"}))
    tool = AlterLabScrapeTool(api_key="sk_test", scrape_params={"mode": "js"})

    assert tool.run(url="https://example.com") == "body"
    assert json.loads(rec.requests[0].content)["mode"] == "js"


def test_scrape_polls_job_when_accepted(serve):
    rec = serve(
        ok({"job_id": "job_1"}, status=202),
        ok({"status": "running"}),
        ok({"status": "succeeded", "result": {"content": {"markdown": "done"}}}),
    )

    assert AlterLabScrapeTool(api_key="sk_test").run(url="https://example.com") == "done"
    assert rec.requests[1].url.path == "/api/v1/jobs/job_1"
    assert len(rec.requests) == 3


def test_scrape_failed_job_raises(serve):
    serve(
        ok({"job_id": "job_1"}, status=202),
        ok({"status": "failed", "error": "blocked"}),
    )

    with pytest.raises(RuntimeError, match="blocked"):
        AlterLabScrapeTool(api_key="sk_test")._run("https://example.com")


def test_api_error_includes_status(serve):
    serve(httpx.Response(401, json={"detail": "bad key"}))

    with pytest.raises(RuntimeError, match="401"):
        AlterLabScrapeTool(api_key="sk_bad")._run("https://example.com")


def test_api_key_read_from_env(serve, monkeypatch):
    monkeypatch.setenv("ALTERLAB_API_KEY", "sk_env")
    rec = serve(ok({"content": "ok"}))

    AlterLabScrapeTool().run(url="https://example.com")

    assert rec.requests[0].headers["X-API-Key"] == "sk_env"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ALTERLAB_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ALTERLAB_API_KEY"):
        AlterLabScrapeTool()._run("https://example.com")


def test_search_formats_results(serve):
    rec = serve(
        ok(
            {
                "results": [
                    {"title": "One", "url": "https://one.test", "snippet": "first"},
                    {"title": "Two", "url": "https://two.test"},
                ]
            }
        )
    )

    output = AlterLabSearchTool(api_key="sk_test", num_results=2).run(query="scraping")

    body = json.loads(rec.requests[0].content)
    assert body == {"query": "scraping", "num_results": 2}
    assert rec.requests[0].url.path == "/api/v1/search"
    assert "1. One\nhttps://one.test\nfirst" in output
    assert "2. Two\nhttps://two.test" in output


def test_search_no_results(serve):
    serve(ok({"results": []}))

    assert "No results" in AlterLabSearchTool(api_key="sk_test").run(query="x")


def test_crawl_polls_and_combines_pages(serve):
    rec = serve(
        ok({"crawl_id": "c1"}),
        ok({"status": "running"}),
        ok(
            {
                "status": "completed",
                "pages": [
                    {"url": "https://a.test", "content": {"markdown": "A"}},
                    {"url": "https://a.test/b", "content": {"markdown": "B"}},
                    {"url": "https://a.test/c", "content": {}},
                ],
            }
        ),
    )

    output = AlterLabCrawlTool(api_key="sk_test", max_pages=3, max_depth=1).run(
        url="https://a.test"
    )

    body = json.loads(rec.requests[0].content)
    assert body["max_pages"] == 3
    assert body["max_depth"] == 1
    assert rec.requests[1].url.path == "/api/v1/crawl/c1"
    assert rec.requests[1].url.params["include_results"] == "true"
    assert output == "## https://a.test\n\nA\n\n---\n\n## https://a.test/b\n\nB"


def test_crawl_without_content(serve):
    serve(ok({"crawl_id": "c1"}), ok({"status": "completed", "pages": []}))

    assert "No content" in AlterLabCrawlTool(api_key="sk_test").run(url="https://a.test")

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Literal
from unittest.mock import Mock, patch

import pytest

from crewai_tools import (
    ContextBrandTool,
    ContextCrawlTool,
    ContextExtractTool,
    ContextParseTool,
    ContextScrapeTool,
    ContextSearchTool,
    ContextSitemapTool,
)


def successful_response(payload: object) -> Mock:
    response = Mock()
    response.ok = True
    response.status_code = 200
    response.text = ""
    response.json.return_value = payload
    return response


@pytest.fixture
def request_mock() -> Generator[Mock, None, None]:
    with patch(
        "crewai_tools.tools.context_dev_tools.base.requests.request"
    ) as mocked_request:
        mocked_request.return_value = successful_response({"ok": True})
        yield mocked_request


def test_context_tools_are_exported_with_described_inputs() -> None:
    tools = [
        ContextSearchTool,
        ContextScrapeTool,
        ContextCrawlTool,
        ContextSitemapTool,
        ContextExtractTool,
        ContextParseTool,
        ContextBrandTool,
    ]

    for tool_class in tools:
        tool = tool_class(api_key="test-key")
        schema = tool.args_schema.model_json_schema()

        assert tool.name.startswith("Context.dev")
        assert tool.description
        assert schema["properties"]
        assert all(
            property_schema.get("description")
            for property_schema in schema["properties"].values()
        )


def test_missing_api_key_fails_before_network_request(
    monkeypatch: pytest.MonkeyPatch,
    request_mock: Mock,
) -> None:
    monkeypatch.delenv("CONTEXT_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CONTEXT_API_KEY"):
        ContextSearchTool()._run(query="current AI news")

    request_mock.assert_not_called()


def test_search_sends_filters_and_markdown_options(request_mock: Mock) -> None:
    result = ContextSearchTool(api_key="test-key")._run(
        query="Stripe product launches",
        num_results=20,
        include_domains=["stripe.com"],
        exclude_domains=["support.stripe.com"],
        freshness="last_month",
        country="us",
        query_fanout=True,
        include_markdown=True,
        timeout_ms=30000,
    )

    assert result == {"ok": True}
    request_mock.assert_called_once_with(
        method="POST",
        url="https://api.context.dev/v1/web/search",
        headers={
            "Authorization": "Bearer test-key",
            "User-Agent": "crewai-tools/context-dev",
        },
        params=[],
        json={
            "query": "Stripe product launches",
            "numResults": 20,
            "includeDomains": ["stripe.com"],
            "excludeDomains": ["support.stripe.com"],
            "freshness": "last_month",
            "country": "us",
            "queryFanout": True,
            "markdownOptions": {
                "enabled": True,
                "useMainContentOnly": True,
            },
            "timeoutMS": 30000,
        },
        timeout=180.0,
    )


def test_scrape_serializes_query_parameters(request_mock: Mock) -> None:
    ContextScrapeTool(api_key="test-key")._run(
        url="https://example.com/pricing",
        include_links=False,
        include_images=True,
        use_main_content_only=True,
        include_html=True,
        include_selectors=["main", ".pricing"],
        exclude_selectors=["nav"],
        max_age_ms=0,
        wait_for_ms=250,
        country="gb",
        timeout_ms=20000,
    )

    request_mock.assert_called_once_with(
        method="GET",
        url="https://api.context.dev/v1/web/scrape/markdown",
        headers={
            "Authorization": "Bearer test-key",
            "User-Agent": "crewai-tools/context-dev",
        },
        params=[
            ("url", "https://example.com/pricing"),
            ("includeLinks", "false"),
            ("includeImages", "true"),
            ("useMainContentOnly", "true"),
            ("includeHTML", "true"),
            ("includeSelectors", "main"),
            ("includeSelectors", ".pricing"),
            ("excludeSelectors", "nav"),
            ("maxAgeMs", "0"),
            ("waitForMs", "250"),
            ("country", "gb"),
            ("timeoutMS", "20000"),
        ],
        timeout=180.0,
    )


def test_crawl_sends_camel_case_body(request_mock: Mock) -> None:
    ContextCrawlTool(api_key="test-key")._run(
        url="https://docs.example.com",
        max_pages=25,
        max_depth=3,
        url_regex="/guides/",
        follow_subdomains=True,
        include_links=True,
        include_images=True,
        use_main_content_only=True,
        include_selectors=["article"],
        exclude_selectors=["aside"],
        stop_after_ms=60000,
        timeout_ms=90000,
    )

    body = request_mock.call_args.kwargs["json"]
    assert body == {
        "url": "https://docs.example.com",
        "maxPages": 25,
        "maxDepth": 3,
        "urlRegex": "/guides/",
        "followSubdomains": True,
        "includeLinks": True,
        "includeImages": True,
        "useMainContentOnly": True,
        "includeSelectors": ["article"],
        "excludeSelectors": ["aside"],
        "stopAfterMs": 60000,
        "timeoutMS": 90000,
    }


def test_sitemap_combines_search_regex_and_explicit_sitemap(request_mock: Mock) -> None:
    ContextSitemapTool(api_key="test-key")._run(
        domain="example.com",
        max_links=50,
        search="API authentication guides",
        url_regex="/docs/",
        sitemap_url="https://example.com/sitemap.xml",
        timeout_ms=45000,
    )

    assert request_mock.call_args.kwargs["params"] == [
        ("domain", "example.com"),
        ("maxLinks", "50"),
        ("search", "API authentication guides"),
        ("urlRegex", "/docs/"),
        ("sitemapUrl", "https://example.com/sitemap.xml"),
        ("timeoutMS", "45000"),
    ]


def test_extract_preserves_schema_without_mutating_it(request_mock: Mock) -> None:
    schema = {
        "type": "object",
        "properties": {"price": {"type": "number"}},
        "required": ["price"],
    }
    original = {
        "type": "object",
        "properties": {"price": {"type": "number"}},
        "required": ["price"],
    }

    ContextExtractTool(api_key="test-key")._run(
        url="https://example.com/pricing",
        response_schema=schema,
        instructions="Return the monthly price.",
        fact_check=True,
        max_pages=3,
    )

    assert schema == original
    assert request_mock.call_args.kwargs["json"]["schema"] == original


@pytest.mark.parametrize(
    ("lookup_type", "identifier", "expected"),
    [
        ("domain", "stripe.com", {"type": "by_domain", "domain": "stripe.com"}),
        ("name", "Stripe", {"type": "by_name", "name": "Stripe"}),
        ("email", "a@stripe.com", {"type": "by_email", "email": "a@stripe.com"}),
        ("ticker", "STRP", {"type": "by_ticker", "ticker": "STRP"}),
        (
            "direct_url",
            "https://stripe.com/payments",
            {"type": "by_direct_url", "direct_url": "https://stripe.com/payments"},
        ),
        (
            "transaction",
            "STRIPE PAYMENTS",
            {"type": "by_transaction", "transaction_info": "STRIPE PAYMENTS"},
        ),
    ],
)
def test_brand_maps_lookup_discriminator(
    lookup_type: Literal[
        "domain", "name", "email", "ticker", "direct_url", "transaction"
    ],
    identifier: str,
    expected: dict[str, str],
    request_mock: Mock,
) -> None:
    ContextBrandTool(api_key="test-key")._run(
        identifier=identifier,
        lookup_type=lookup_type,
    )

    body = request_mock.call_args.kwargs["json"]
    assert body == {**expected, **({} if lookup_type == "direct_url" else {"maxSpeed": False})}


def test_parse_uploads_local_bytes_with_pdf_options(
    tmp_path: Path,
    request_mock: Mock,
) -> None:
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"%PDF-test")

    result = ContextParseTool(api_key="test-key", base_dir=str(tmp_path))._run(
        file_path="sample.pdf",
        include_images=True,
        ocr=True,
        pdf_start=2,
        pdf_end=4,
    )

    assert result == {"ok": True}
    request_mock.assert_called_once_with(
        method="POST",
        url="https://api.context.dev/v1/parse",
        headers={
            "Authorization": "Bearer test-key",
            "User-Agent": "crewai-tools/context-dev",
            "Content-Type": "application/octet-stream",
        },
        params=[
            ("extension", "pdf"),
            ("includeLinks", "true"),
            ("includeImages", "true"),
            ("useMainContentOnly", "false"),
            ("ocr", "true"),
            ("pdf[start]", "2"),
            ("pdf[end]", "4"),
        ],
        data=b"%PDF-test",
        timeout=180.0,
    )


def test_parse_rejects_invalid_page_range(tmp_path: Path, request_mock: Mock) -> None:
    document = tmp_path / "sample.pdf"
    document.write_bytes(b"%PDF-test")

    with pytest.raises(ValueError, match="pdf_end"):
        ContextParseTool(api_key="test-key", base_dir=str(tmp_path))._run(
            file_path="sample.pdf",
            pdf_start=4,
            pdf_end=2,
        )

    request_mock.assert_not_called()


def test_api_error_uses_server_message(request_mock: Mock) -> None:
    response = Mock()
    response.ok = False
    response.status_code = 422
    response.text = ""
    response.json.return_value = {"error": "Invalid URL"}
    request_mock.return_value = response

    with pytest.raises(RuntimeError, match="Context.dev API 422: Invalid URL"):
        ContextScrapeTool(api_key="test-key")._run(url="not-a-url")


def test_internal_metadata_is_removed_from_response(request_mock: Mock) -> None:
    request_mock.return_value = successful_response(
        {
            "data": {"title": "Example"},
            "request_id": "req-secret",
            "key_metadata": {"credits": 1},
        }
    )

    result = ContextScrapeTool(api_key="test-key")._run(url="https://example.com")

    assert result == {"data": {"title": "Example"}}

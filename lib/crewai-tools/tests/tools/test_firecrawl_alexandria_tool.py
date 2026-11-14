from unittest.mock import MagicMock, patch

import pytest
from crewai.tools.tool_failure import ToolFailure
from firecrawl.v2.utils.error_handler import FirecrawlError
from firecrawl.v2.types import (
    AlexandriaError,
    AlexandriaScrapeData,
    AlexandriaScrapeResult,
)

from crewai_tools.tools.firecrawl_alexandria_tool.firecrawl_alexandria_tool import (
    FirecrawlAlexandriaTool,
)


@pytest.fixture
def firecrawl_client():
    with patch("firecrawl.Firecrawl") as client_cls:
        yield client_cls.return_value


def test_run_returns_capability_data(firecrawl_client: MagicMock) -> None:
    firecrawl_client.scrape.return_value = AlexandriaScrapeData(
        alexandria=[
            AlexandriaScrapeResult(
                provider="sec-gov",
                capability="filings/companies_get",
                data={"record": {"cik": "0000320193"}},
            )
        ]
    )
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    result = tool.run(
        provider="sec-gov",
        capability="filings/companies_get",
        options='{"cik": "320193"}',
    )

    assert result == {"record": {"cik": "0000320193"}}
    firecrawl_client.scrape.assert_called_once_with(
        alexandria={
            "provider": "sec-gov",
            "capability": "filings/companies_get",
            "options": {"cik": "320193"},
        },
        integration="_crewai",
    )


def test_run_without_options_sends_empty_options(firecrawl_client: MagicMock) -> None:
    firecrawl_client.scrape.return_value = AlexandriaScrapeData(
        alexandria=[AlexandriaScrapeResult(data={"ok": True})]
    )
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    tool.run(provider="sec-gov", capability="filings/companies_status")

    call = firecrawl_client.scrape.call_args
    assert call.kwargs["alexandria"]["options"] == {}


def test_run_reports_provider_error_as_failure(firecrawl_client: MagicMock) -> None:
    firecrawl_client.scrape.return_value = AlexandriaScrapeData(
        alexandria=[
            AlexandriaScrapeResult(
                error=AlexandriaError(
                    code="missing_option",
                    message="filings/companies_get requires cik.",
                    status=400,
                )
            )
        ]
    )
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    result = tool._run(provider="sec-gov", capability="filings/companies_get")

    assert isinstance(result, ToolFailure)
    assert result.code == "missing_option"
    assert "requires cik" in result.message
    assert result.retryable is False


def test_run_marks_upstream_errors_retryable(firecrawl_client: MagicMock) -> None:
    firecrawl_client.scrape.return_value = AlexandriaScrapeData(
        alexandria=[
            AlexandriaScrapeResult(
                error=AlexandriaError(
                    code="upstream_error", message="Provider timed out.", status=503
                )
            )
        ]
    )
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    result = tool._run(provider="sec-gov", capability="filings/companies_get")

    assert isinstance(result, ToolFailure)
    assert result.retryable is True


@pytest.mark.parametrize("options", ["{not json", "[1, 2]"])
def test_run_rejects_options_that_are_not_a_json_object(
    firecrawl_client: MagicMock, options: str
) -> None:
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    result = tool._run(
        provider="sec-gov", capability="filings/companies_get", options=options
    )

    assert isinstance(result, ToolFailure)
    assert result.code == "invalid_options"
    firecrawl_client.scrape.assert_not_called()


def test_run_reports_api_errors_as_failure(firecrawl_client: MagicMock) -> None:
    firecrawl_client.scrape.side_effect = FirecrawlError(
        "Unknown capability. Did you mean: filings/company",
        404,
        code="unknown_capability",
    )
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    result = tool._run(provider="sec-gov", capability="filings/companie")

    assert isinstance(result, ToolFailure)
    assert result.code == "unknown_capability"
    assert "Did you mean: filings/company" in result.message
    assert result.retryable is False


def test_run_reports_empty_response_as_failure(firecrawl_client: MagicMock) -> None:
    firecrawl_client.scrape.return_value = AlexandriaScrapeData(alexandria=[])
    tool = FirecrawlAlexandriaTool(api_key="test-key")

    result = tool._run(provider="sec-gov", capability="filings/companies_get")

    assert isinstance(result, ToolFailure)
    assert result.code == "empty_response"


@pytest.mark.vcr()
def test_firecrawl_alexandria_tool_integration() -> None:
    tool = FirecrawlAlexandriaTool()
    result = tool.run(
        provider="sec-gov",
        capability="filings/companies_get",
        options='{"cik": "320193"}',
    )

    assert isinstance(result, dict)
    assert result["record"]["cik"] == "0000320193"

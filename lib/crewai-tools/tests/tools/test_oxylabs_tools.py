from collections.abc import Callable
import json
import logging
import os
import threading
from unittest.mock import MagicMock

from crewai.tools.base_tool import BaseTool
from crewai.tools.tool_failure import ToolFailure
from crewai_tools import (
    OxylabsAmazonProductScraperTool,
    OxylabsAmazonSearchScraperTool,
    OxylabsGoogleSearchScraperTool,
    OxylabsUniversalScraperTool,
)
from crewai_tools.tools.oxylabs_amazon_product_scraper_tool.oxylabs_amazon_product_scraper_tool import (
    OxylabsAmazonProductScraperConfig,
)
from crewai_tools.tools.oxylabs_base_tool.oxylabs_base_tool import OxylabsBaseTool
from crewai_tools.tools.oxylabs_google_search_scraper_tool.oxylabs_google_search_scraper_tool import (
    OxylabsGoogleSearchScraperConfig,
)
from crewai_tools.tools.oxylabs_universal_scraper_tool.oxylabs_universal_scraper_tool import (
    OxylabsUniversalScraperArgs,
)
from oxylabs import RealtimeClient
from oxylabs.sources.response import Response as OxylabsResponse
from pydantic import BaseModel
import pytest


@pytest.fixture
def oxylabs_api() -> RealtimeClient:
    oxylabs_api_mock = MagicMock()

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Scraping Sandbox</title>
    </head>
    <body>
    <div id="main">
        <div id="product-list">
            <div>
                <p>Amazing product</p>
                <p>Price $14.99</p>
            </div>
            <div>
                <p>Good product</p>
                <p>Price $9.99</p>
            </div>
        </div>
    </div>
    </body>
    </html>
    """

    json_content = {
        "results": {
            "products": [
                {"title": "Amazing product", "price": 14.99, "currency": "USD"},
                {"title": "Good product", "price": 9.99, "currency": "USD"},
            ],
        },
    }

    html_response = OxylabsResponse({"results": [{"content": html_content}]})
    json_response = OxylabsResponse({"results": [{"content": json_content}]})

    oxylabs_api_mock.universal.scrape_url.side_effect = [json_response, html_response]
    oxylabs_api_mock.amazon.scrape_search.side_effect = [json_response, html_response]
    oxylabs_api_mock.amazon.scrape_product.side_effect = [json_response, html_response]
    oxylabs_api_mock.google.scrape_search.side_effect = [json_response, html_response]

    return oxylabs_api_mock


@pytest.mark.parametrize(
    ("tool_class",),
    [
        (OxylabsUniversalScraperTool,),
        (OxylabsAmazonSearchScraperTool,),
        (OxylabsGoogleSearchScraperTool,),
        (OxylabsAmazonProductScraperTool,),
    ],
)
def test_tool_initialization(tool_class: type[BaseTool]):
    tool = tool_class(username="username", password="password")
    assert isinstance(tool, tool_class)


@pytest.mark.parametrize(
    ("tool_class",),
    [
        (OxylabsUniversalScraperTool,),
        (OxylabsAmazonSearchScraperTool,),
        (OxylabsGoogleSearchScraperTool,),
        (OxylabsAmazonProductScraperTool,),
    ],
)
def test_tool_initialization_with_env_vars(tool_class: type[BaseTool]):
    os.environ["OXYLABS_USERNAME"] = "username"
    os.environ["OXYLABS_PASSWORD"] = "password"

    tool = tool_class()
    assert isinstance(tool, tool_class)

    del os.environ["OXYLABS_USERNAME"]
    del os.environ["OXYLABS_PASSWORD"]


@pytest.mark.parametrize(
    ("tool_class",),
    [
        (OxylabsUniversalScraperTool,),
        (OxylabsAmazonSearchScraperTool,),
        (OxylabsGoogleSearchScraperTool,),
        (OxylabsAmazonProductScraperTool,),
    ],
)
def test_tool_initialization_failure(tool_class: type[BaseTool]):
    for key in ["OXYLABS_USERNAME", "OXYLABS_PASSWORD"]:
        if key in os.environ:
            del os.environ[key]

    with pytest.raises(ValueError):
        tool_class()


@pytest.mark.parametrize(
    ("tool_class", "tool_config"),
    [
        (OxylabsUniversalScraperTool, {"geo_location": "Paris, France"}),
        (
            OxylabsAmazonSearchScraperTool,
            {"domain": "co.uk"},
        ),
        (
            OxylabsGoogleSearchScraperTool,
            OxylabsGoogleSearchScraperConfig(render="html"),
        ),
        (
            OxylabsAmazonProductScraperTool,
            OxylabsAmazonProductScraperConfig(parse=True),
        ),
    ],
)
def test_tool_invocation(
    tool_class: type[BaseTool],
    tool_config: BaseModel,
    oxylabs_api: RealtimeClient,
):
    tool = tool_class(username="username", password="password", config=tool_config)

    # setting via __dict__ to bypass pydantic validation
    tool.__dict__["oxylabs_api"] = oxylabs_api

    result = tool.run("Scraping Query 1")
    assert isinstance(result, str)
    assert isinstance(json.loads(result), dict)

    result = tool.run("Scraping Query 2")
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result


ALL_TOOL_CLASSES = [
    OxylabsUniversalScraperTool,
    OxylabsAmazonSearchScraperTool,
    OxylabsGoogleSearchScraperTool,
    OxylabsAmazonProductScraperTool,
]


def build_tool(
    tool_class: type[BaseTool],
    raw_response: dict,
    sdk_logs: list[str] | None = None,
    config: BaseModel | None = None,
) -> BaseTool:
    """Build a tool whose every scrape entrypoint answers with ``raw_response``.

    ``sdk_logs`` reproduces the oxylabs SDK's habit of logging the real cause and
    returning an empty response instead of raising.
    """
    api = MagicMock()
    response = OxylabsResponse(raw_response)

    def scrape(*_args: object, **_kwargs: object) -> OxylabsResponse:
        for line in sdk_logs or []:
            logging.getLogger("oxylabs.internal.api").error(line)
        return response

    api.universal.scrape_url.side_effect = scrape
    api.amazon.scrape_search.side_effect = scrape
    api.amazon.scrape_product.side_effect = scrape
    api.google.scrape_search.side_effect = scrape

    tool = tool_class(username="username", password="password", config=config)
    # setting via __dict__ to bypass pydantic validation
    tool.__dict__["oxylabs_api"] = api
    return tool


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_rejected_request_reports_failure(tool_class: type[BaseTool]):
    """The SDK logs HTTP errors and returns an empty response instead of raising,
    so a rejected request must be reported rather than indexed into."""
    result = build_tool(tool_class, {}).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == "empty_response"
    assert "OXYLABS_USERNAME" in result.message


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(404, False), (429, True), (500, True), (503, True)],
)
def test_upstream_error_status_reports_failure(
    tool_class: type[BaseTool], status_code: int, retryable: bool
):
    """A non-2xx result carries no page; returning its empty content would hand
    the agent '[]' as though the scrape had succeeded."""
    result = build_tool(
        tool_class, {"results": [{"content": [], "status_code": status_code}]}
    ).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == str(status_code)
    assert result.retryable is retryable


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_missing_content_reports_failure(tool_class: type[BaseTool]):
    result = build_tool(
        tool_class, {"results": [{"content": None, "status_code": 200}]}
    ).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == "empty_content"


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_list_content_is_serialized_as_json(tool_class: type[BaseTool]):
    """``parsing_instructions`` can yield a list; str() on it would produce a
    Python repr with single quotes rather than JSON."""
    result = build_tool(
        tool_class,
        {"results": [{"content": [{"title": "Amazing product"}], "status_code": 200}]},
    ).run("Scraping Query")

    assert isinstance(result, str)
    assert json.loads(result) == [{"title": "Amazing product"}]


def test_subclass_without_config_field_is_reported():
    """The base class defaults ``config`` from the subclass's own model, so a
    subclass that declares none must say so rather than raise ``KeyError``."""

    class MissingConfig(OxylabsBaseTool):
        name: str = "missing config"
        description: str = "declares no config field"
        args_schema: type[BaseModel] = OxylabsUniversalScraperArgs

        def _run(self, url: str) -> str:
            return ""

    with pytest.raises(TypeError, match="must declare a 'config' model field"):
        MissingConfig(username="username", password="password")


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_rejected_request_names_the_http_cause(tool_class: type[BaseTool]):
    """The agent cannot act on "go read a log", so the status the SDK logged has
    to reach the failure itself."""
    result = build_tool(
        tool_class,
        {},
        sdk_logs=[
            "HTTP error occurred: 401 Client Error: Unauthorized for url: "
            "https://realtime.oxylabs.io/v1/queries",
            "",
        ],
    ).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == "401"
    assert "401 Unauthorized" in result.message
    assert result.retryable is False


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_rejected_request_includes_the_api_explanation(tool_class: type[BaseTool]):
    """The API explains config it rejects; that explanation is what makes the
    failure actionable."""
    result = build_tool(
        tool_class,
        {},
        sdk_logs=[
            "HTTP error occurred: 400 Client Error: Bad Request for url: "
            "https://realtime.oxylabs.io/v1/queries",
            '{"message": "Parameter `parsing_instructions` can be used just with '
            '`parse` parameter set to `true`."}',
        ],
    ).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == "400"
    assert "400 Bad Request" in result.message
    assert "`parsing_instructions` can be used just with" in result.message


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_timeout_is_reported_as_retryable(tool_class: type[BaseTool]):
    result = build_tool(
        tool_class,
        {},
        sdk_logs=[
            "Timeout error. The request to https://realtime.oxylabs.io/v1/queries "
            "with method POST has timed out."
        ],
    ).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == "timeout"
    assert result.retryable is True


@pytest.mark.parametrize("tool_class", ALL_TOOL_CLASSES)
def test_server_error_is_reported_as_retryable(tool_class: type[BaseTool]):
    result = build_tool(
        tool_class,
        {},
        sdk_logs=[
            "HTTP error occurred: 502 Server Error: Bad Gateway for url: "
            "https://realtime.oxylabs.io/v1/queries"
        ],
    ).run("Scraping Query")

    assert isinstance(result, ToolFailure)
    assert result.code == "502"
    assert result.retryable is True


def test_google_config_forwards_locale():
    """`locale` is a documented Google Search parameter; the config model used to
    omit it, so it was silently dropped."""
    tool = build_tool(
        OxylabsGoogleSearchScraperTool,
        {"results": [{"content": {"ok": True}, "status_code": 200}]},
        config=OxylabsGoogleSearchScraperConfig(locale="de", limit=2),
    )

    tool.run("iPhone 16")

    _, kwargs = tool.oxylabs_api.google.scrape_search.call_args
    assert kwargs["locale"] == "de"
    assert kwargs["limit"] == 2


def test_result_without_content_is_reported():
    """A result object missing `content` entirely must not raise AttributeError."""
    tool = build_tool(OxylabsUniversalScraperTool, {"results": [{"status_code": 200}]})

    result = tool.run("https://example.com")

    assert isinstance(result, ToolFailure)
    assert result.code == "empty_content"


def test_concurrent_scrapes_do_not_share_diagnoses():
    """Two scrapes in flight at once must each be diagnosed from their own error.

    A shared collector would hand both calls both errors, and the timeout below
    would be reported as the other request's non-retryable 400 -- telling the
    agent not to retry something it should.
    """
    both_started = threading.Barrier(2)
    timeout_logged = threading.Event()
    bad_request_logged = threading.Event()
    outcomes: dict[str, ToolFailure] = {}

    def tool_logging(emit: Callable[[], None]) -> BaseTool:
        api = MagicMock()

        def scrape(*_args: object, **_kwargs: object) -> OxylabsResponse:
            both_started.wait(timeout=5)
            emit()
            return OxylabsResponse({})

        api.universal.scrape_url.side_effect = scrape
        tool = OxylabsUniversalScraperTool(username="username", password="password")
        tool.__dict__["oxylabs_api"] = api
        return tool

    sdk_logger = logging.getLogger("oxylabs.internal.api")

    def emit_timeout() -> None:
        sdk_logger.error(
            "Timeout error. The request to https://realtime.oxylabs.io/v1/queries "
            "with method POST has timed out."
        )
        timeout_logged.set()
        # Hold this capture open while the other call logs, which is the window
        # in which the two could bleed into each other.
        bad_request_logged.wait(timeout=5)

    def emit_bad_request() -> None:
        timeout_logged.wait(timeout=5)
        sdk_logger.error(
            "HTTP error occurred: 400 Client Error: Bad Request for url: "
            "https://realtime.oxylabs.io/v1/queries"
        )
        bad_request_logged.set()

    def run(key: str, emit: Callable[[], None]) -> None:
        outcomes[key] = tool_logging(emit).run("https://example.com")

    threads = [
        threading.Thread(target=run, args=("timeout", emit_timeout)),
        threading.Thread(target=run, args=("bad_request", emit_bad_request)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert outcomes["timeout"].code == "timeout"
    assert outcomes["timeout"].retryable is True
    assert outcomes["bad_request"].code == "400"
    assert outcomes["bad_request"].retryable is False

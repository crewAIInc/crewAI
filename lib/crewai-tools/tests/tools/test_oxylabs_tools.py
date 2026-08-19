import json
import os
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
from crewai_tools.tools.oxylabs_google_search_scraper_tool.oxylabs_google_search_scraper_tool import (
    OxylabsGoogleSearchScraperConfig,
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


def build_tool(tool_class: type[BaseTool], raw_response: dict) -> BaseTool:
    """Build a tool whose every scrape entrypoint answers with ``raw_response``."""
    api = MagicMock()
    response = OxylabsResponse(raw_response)
    api.universal.scrape_url.return_value = response
    api.amazon.scrape_search.return_value = response
    api.amazon.scrape_product.return_value = response
    api.google.scrape_search.return_value = response

    tool = tool_class(username="username", password="password")
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
    [(404, False), (429, True), (503, True)],
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

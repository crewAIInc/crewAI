import json
import os
from unittest.mock import MagicMock

from crewai.tools.base_tool import BaseTool
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
    # making sure env vars are not set
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

    # verifying parsed job returns json content
    result = tool.run("Scraping Query 1")
    assert isinstance(result, str)
    assert isinstance(json.loads(result), dict)

    # verifying raw job returns str
    result = tool.run("Scraping Query 2")
    assert isinstance(result, str)
    assert "<!DOCTYPE html>" in result

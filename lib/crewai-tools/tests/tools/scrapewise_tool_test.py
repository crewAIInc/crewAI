import sys
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.scrapewise_tool.scrapewise_tool import (
    FixedScrapewiseProductDataToolSchema,
    ScrapewiseProductDataTool,
    ScrapewiseProductDataToolSchema,
)


SAMPLE_ROWS = [
    {
        "title": "4005808655267",
        "url": "https://shop.example/p/1",
        "name": "Nivea Soft 200ml",
        "price": "4.99",
        "currency": "EUR",
        "availability": "InStock",
        "_sw_scraper": "65f1",
        "_sw_run_date": "2026-10-07",
    },
    {
        "title": "4005808655274",
        "url": "https://shop.example/p/2",
        "name": "Nivea Soft 100ml",
        "price": "3.49",
        "currency": "EUR",
        "availability": "OutOfStock",
        "_sw_scraper": "65f1",
        "_sw_run_date": "2026-10-07",
    },
]


@pytest.fixture(autouse=True)
def stub_scrapewise_sdk():
    """Stub the SDK so the suite runs without the optional `scrapewise` extra.

    `scrapewise` is an optional dependency, so it is absent from a default
    install. Every test here mocks the client anyway, so the real package is
    never needed -- stubbing the module keeps the whole suite running either
    way, rather than skipping it when the extra is not installed.
    """
    module = MagicMock()
    module.ScrapewiseClient = MagicMock()
    with patch.dict(sys.modules, {"scrapewise": module}):
        yield module


def initialize_tool_with(mock_client, scraper_id=None):
    with patch.dict("os.environ", {"SCRAPEWISE_API_KEY": "test_api_key"}):
        tool = ScrapewiseProductDataTool(scraper_id=scraper_id)
    tool._client = mock_client
    return tool


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def tool(mock_client):
    return initialize_tool_with(mock_client)


def test_tool_initialization(tool):
    assert tool.name == "ScrapeWise product data reader"
    assert "ScrapeWise" in tool.description
    assert tool.args_schema is ScrapewiseProductDataToolSchema
    assert tool.scraper_id is None
    assert tool.package_dependencies == ["scrapewise"]


def test_tool_declares_its_required_env_var(tool):
    env_var = next(v for v in tool.env_vars if v.name == "SCRAPEWISE_API_KEY")
    assert env_var.required is True


@patch.dict("os.environ", {}, clear=True)
def test_tool_initialization_without_api_key_raises():
    with pytest.raises(ValueError, match="ScrapeWise API key is required"):
        ScrapewiseProductDataTool()


@patch.dict("os.environ", {"SCRAPEWISE_API_KEY": "test_api_key"})
def test_tool_initialization_without_the_package_raises_import_error():
    with (
        patch.dict(sys.modules, {"scrapewise": None}),
        pytest.raises(ImportError, match="uv add scrapewise"),
    ):
        ScrapewiseProductDataTool()


@patch.dict("os.environ", {}, clear=True)
def test_tool_initialization_with_explicit_api_key():
    tool = ScrapewiseProductDataTool(api_key="explicit_key")

    assert tool.api_key == "explicit_key"


@patch.dict("os.environ", {"SCRAPEWISE_API_KEY": "test_api_key"})
def test_tool_initialization_with_fixed_scraper_id():
    tool = ScrapewiseProductDataTool(scraper_id="65f1")

    assert tool.scraper_id == "65f1"
    assert tool.args_schema is FixedScrapewiseProductDataToolSchema
    assert "65f1" in tool.description
    assert "scraper_id" not in FixedScrapewiseProductDataToolSchema.model_fields


@patch.dict("os.environ", {"SCRAPEWISE_API_KEY": "test_api_key"})
def test_tool_initialization_with_base_url_and_timeout():
    tool = ScrapewiseProductDataTool(
        base_url="https://staging.example/api", timeout=120.0
    )

    assert tool.base_url == "https://staging.example/api"
    assert tool.timeout == 120.0


def test_run_returns_formatted_rows(tool, mock_client):
    mock_client.get_sample_data.return_value = SAMPLE_ROWS

    result = tool._run(scraper_id="65f1")

    mock_client.get_sample_data.assert_called_once_with("65f1")
    assert isinstance(result, str)
    assert "2 product row(s) from ScrapeWise:" in result
    assert "Nivea Soft 200ml" in result
    assert "price: 4.99" in result


def test_run_strips_internal_columns(tool, mock_client):
    mock_client.get_sample_data.return_value = SAMPLE_ROWS

    result = tool._run(scraper_id="65f1")

    assert "_sw_scraper" not in result
    assert "_sw_run_date" not in result


def test_run_respects_max_rows(tool, mock_client):
    mock_client.get_sample_data.return_value = SAMPLE_ROWS

    result = tool._run(scraper_id="65f1", max_rows=1)

    assert "1 product row(s)" in result
    assert "Nivea Soft 200ml" in result
    assert "Nivea Soft 100ml" not in result


def test_run_unwraps_paginated_envelope(tool, mock_client):
    mock_client.get_sample_data.return_value = {"content": SAMPLE_ROWS}

    result = tool._run(scraper_id="65f1")

    assert "2 product row(s)" in result


def test_run_reports_no_stored_rows(tool, mock_client):
    mock_client.get_sample_data.return_value = []

    result = tool._run(scraper_id="65f1")

    assert "No product rows stored" in result


def test_run_without_scraper_id_returns_error(tool, mock_client):
    result = tool._run()

    assert "Error" in result
    assert "scraper_id" in result
    mock_client.get_sample_data.assert_not_called()


def test_run_uses_fixed_scraper_id(mock_client):
    tool = initialize_tool_with(mock_client, scraper_id="65f1")
    mock_client.get_sample_data.return_value = SAMPLE_ROWS

    tool._run()

    mock_client.get_sample_data.assert_called_once_with("65f1")


def test_run_with_api_exception_returns_error(tool, mock_client):
    mock_client.get_sample_data.side_effect = Exception("404 scraper not found")

    result = tool._run(scraper_id="nope")

    assert "Error reading ScrapeWise product data" in result
    assert "404 scraper not found" in result


def test_run_with_unexpected_payload_returns_error(tool, mock_client):
    mock_client.get_sample_data.return_value = "not a list"

    result = tool._run(scraper_id="65f1")

    assert "Error" in result


def test_schema_requires_scraper_id():
    with pytest.raises(ValueError):
        ScrapewiseProductDataToolSchema()


def test_schema_defaults_max_rows_to_25():
    assert ScrapewiseProductDataToolSchema(scraper_id="65f1").max_rows == 25


@pytest.mark.parametrize("max_rows", [0, -1, 101, 1000])
def test_schema_rejects_out_of_range_max_rows(max_rows):
    with pytest.raises(ValueError, match="max_rows must be between 1 and 100"):
        ScrapewiseProductDataToolSchema(scraper_id="65f1", max_rows=max_rows)


@pytest.mark.parametrize("max_rows", [1, 25, 100])
def test_schema_accepts_in_range_max_rows(max_rows):
    schema = ScrapewiseProductDataToolSchema(scraper_id="65f1", max_rows=max_rows)

    assert schema.max_rows == max_rows


def test_fixed_schema_also_validates_max_rows():
    with pytest.raises(ValueError, match="max_rows must be between 1 and 100"):
        FixedScrapewiseProductDataToolSchema(max_rows=101)

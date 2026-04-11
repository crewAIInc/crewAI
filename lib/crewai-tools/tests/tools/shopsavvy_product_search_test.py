import json
import os
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.shopsavvy_product_search.shopsavvy_product_search import (
    ShopSavvyProductSearchTool,
)


@pytest.fixture(autouse=True)
def mock_shopsavvy_api_key():
    with patch.dict(os.environ, {"SHOPSAVVY_API_KEY": "test_key_123"}):
        yield


@pytest.fixture
def tool():
    return ShopSavvyProductSearchTool()


def test_initialization():
    tool = ShopSavvyProductSearchTool()
    assert tool.name == "ShopSavvy Product Search"
    assert tool.api_key == "test_key_123"
    assert tool.base_url == "https://api.shopsavvy.com/v1"


def test_initialization_with_custom_api_key():
    tool = ShopSavvyProductSearchTool(api_key="custom_key")
    assert tool.api_key == "custom_key"


def test_missing_api_key_raises_error():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("SHOPSAVVY_API_KEY", None)
        tool = ShopSavvyProductSearchTool(api_key=None)
        with pytest.raises(ValueError, match="ShopSavvy API key is required"):
            tool.run(query="test product")


@patch("crewai_tools.tools.shopsavvy_product_search.shopsavvy_product_search.requests")
def test_successful_search(mock_requests_module, tool):
    mock_response_data = {
        "products": [
            {
                "id": "abc123",
                "title": "Sony WH-1000XM5",
                "brand": "Sony",
                "offers": [
                    {
                        "retailer": "Amazon",
                        "price": 278.00,
                        "currency": "USD",
                        "url": "https://amazon.com/dp/B09XS7JWHH",
                    },
                    {
                        "retailer": "Best Buy",
                        "price": 299.99,
                        "currency": "USD",
                        "url": "https://bestbuy.com/product/123",
                    },
                ],
            }
        ]
    }
    mock_response = MagicMock()
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status.return_value = None
    mock_requests_module.get.return_value = mock_response

    result = tool.run(query="Sony WH-1000XM5")

    mock_requests_module.get.assert_called_once_with(
        "https://api.shopsavvy.com/v1/products/search",
        params={"query": "Sony WH-1000XM5"},
        headers={
            "Authorization": "Bearer test_key_123",
            "Content-Type": "application/json",
        },
        timeout=30,
    )

    parsed = json.loads(result)
    assert "products" in parsed
    assert len(parsed["products"]) == 1
    assert parsed["products"][0]["title"] == "Sony WH-1000XM5"
    assert len(parsed["products"][0]["offers"]) == 2


@patch("crewai_tools.tools.shopsavvy_product_search.shopsavvy_product_search.requests")
def test_search_with_barcode(mock_requests_module, tool):
    mock_response = MagicMock()
    mock_response.json.return_value = {"products": []}
    mock_response.raise_for_status.return_value = None
    mock_requests_module.get.return_value = mock_response

    tool.run(query="027242923799")

    mock_requests_module.get.assert_called_once_with(
        "https://api.shopsavvy.com/v1/products/search",
        params={"query": "027242923799"},
        headers={
            "Authorization": "Bearer test_key_123",
            "Content-Type": "application/json",
        },
        timeout=30,
    )


@patch("crewai_tools.tools.shopsavvy_product_search.shopsavvy_product_search.requests")
def test_api_error_propagates(mock_requests_module, tool):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("403 Forbidden")
    mock_requests_module.get.return_value = mock_response

    with pytest.raises(Exception, match="403 Forbidden"):
        tool.run(query="test")


if __name__ == "__main__":
    pytest.main([__file__])

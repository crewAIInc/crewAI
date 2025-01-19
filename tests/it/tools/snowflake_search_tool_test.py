import asyncio
import json
from decimal import Decimal

import pytest
from snowflake.connector.errors import DatabaseError, OperationalError

from crewai_tools import SnowflakeConfig, SnowflakeSearchTool

# Test Data
MENU_ITEMS = [
    (10001, "Ice Cream", "Freezing Point", "Lemonade", "Beverage", "Cold Option", 1, 4),
    (
        10002,
        "Ice Cream",
        "Freezing Point",
        "Vanilla Ice Cream",
        "Dessert",
        "Ice Cream",
        2,
        6,
    ),
]

INVALID_QUERIES = [
    ("SELECT * FROM nonexistent_table", "relation 'nonexistent_table' does not exist"),
    ("SELECT invalid_column FROM menu", "invalid identifier 'invalid_column'"),
    ("INVALID SQL QUERY", "SQL compilation error"),
]


# Integration Test Fixtures
@pytest.fixture
def config():
    """Create a Snowflake configuration with test credentials."""
    return SnowflakeConfig(
        account="lwyhjun-wx11931",
        user="crewgitci",
        password="crewaiT00ls_publicCIpass123",
        warehouse="COMPUTE_WH",
        database="tasty_bytes_sample_data",
        snowflake_schema="raw_pos",
    )


@pytest.fixture
def snowflake_tool(config):
    """Create a SnowflakeSearchTool instance."""
    return SnowflakeSearchTool(config=config)


# Integration Tests with Real Snowflake Connection
@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "menu_id,expected_type,brand,item_name,category,subcategory,cost,price", MENU_ITEMS
)
async def test_menu_items(
    snowflake_tool,
    menu_id,
    expected_type,
    brand,
    item_name,
    category,
    subcategory,
    cost,
    price,
):
    """Test menu items with parameterized data for multiple test cases."""
    results = await snowflake_tool._run(
        query=f"SELECT * FROM menu WHERE menu_id = {menu_id}"
    )
    assert len(results) == 1
    menu_item = results[0]

    # Validate all fields
    assert menu_item["MENU_ID"] == menu_id
    assert menu_item["MENU_TYPE"] == expected_type
    assert menu_item["TRUCK_BRAND_NAME"] == brand
    assert menu_item["MENU_ITEM_NAME"] == item_name
    assert menu_item["ITEM_CATEGORY"] == category
    assert menu_item["ITEM_SUBCATEGORY"] == subcategory
    assert menu_item["COST_OF_GOODS_USD"] == cost
    assert menu_item["SALE_PRICE_USD"] == price

    # Validate health metrics JSON structure
    health_metrics = json.loads(menu_item["MENU_ITEM_HEALTH_METRICS_OBJ"])
    assert "menu_item_health_metrics" in health_metrics
    metrics = health_metrics["menu_item_health_metrics"][0]
    assert "ingredients" in metrics
    assert isinstance(metrics["ingredients"], list)
    assert all(isinstance(ingredient, str) for ingredient in metrics["ingredients"])
    assert metrics["is_dairy_free_flag"] in ["Y", "N"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_menu_categories_aggregation(snowflake_tool):
    """Test complex aggregation query on menu categories with detailed validations."""
    results = await snowflake_tool._run(
        query="""
        SELECT 
            item_category,
            COUNT(*) as item_count,
            AVG(sale_price_usd) as avg_price,
            SUM(sale_price_usd - cost_of_goods_usd) as total_margin,
            COUNT(DISTINCT menu_type) as menu_type_count,
            MIN(sale_price_usd) as min_price,
            MAX(sale_price_usd) as max_price
        FROM menu
        GROUP BY item_category
        HAVING COUNT(*) > 1
        ORDER BY item_count DESC
        """
    )

    assert len(results) > 0
    for category in results:
        # Basic presence checks
        assert all(
            key in category
            for key in [
                "ITEM_CATEGORY",
                "ITEM_COUNT",
                "AVG_PRICE",
                "TOTAL_MARGIN",
                "MENU_TYPE_COUNT",
                "MIN_PRICE",
                "MAX_PRICE",
            ]
        )

        # Value validations
        assert category["ITEM_COUNT"] > 1  # Due to HAVING clause
        assert category["MIN_PRICE"] <= category["MAX_PRICE"]
        assert category["AVG_PRICE"] >= category["MIN_PRICE"]
        assert category["AVG_PRICE"] <= category["MAX_PRICE"]
        assert category["MENU_TYPE_COUNT"] >= 1
        assert isinstance(category["TOTAL_MARGIN"], (float, Decimal))


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_query,expected_error", INVALID_QUERIES)
async def test_invalid_queries(snowflake_tool, invalid_query, expected_error):
    """Test error handling for invalid queries."""
    with pytest.raises((DatabaseError, OperationalError)) as exc_info:
        await snowflake_tool._run(query=invalid_query)
    assert expected_error.lower() in str(exc_info.value).lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_queries(snowflake_tool):
    """Test handling of concurrent queries."""
    queries = [
        "SELECT COUNT(*) FROM menu",
        "SELECT COUNT(DISTINCT menu_type) FROM menu",
        "SELECT COUNT(DISTINCT item_category) FROM menu",
    ]

    tasks = [snowflake_tool._run(query=query) for query in queries]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(isinstance(result, list) for result in results)
    assert all(len(result) == 1 for result in results)
    assert all(isinstance(result[0], dict) for result in results)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_timeout(snowflake_tool):
    """Test query timeout handling with a complex query."""
    with pytest.raises((DatabaseError, OperationalError)) as exc_info:
        await snowflake_tool._run(
            query="""
            WITH RECURSIVE numbers AS (
                SELECT 1 as n
                UNION ALL
                SELECT n + 1
                FROM numbers
                WHERE n < 1000000
            )
            SELECT COUNT(*) FROM numbers
            """
        )
    assert (
        "timeout" in str(exc_info.value).lower()
        or "execution time" in str(exc_info.value).lower()
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_caching_behavior(snowflake_tool):
    """Test query caching behavior and performance."""
    query = "SELECT * FROM menu LIMIT 5"

    # First execution
    start_time = asyncio.get_event_loop().time()
    results1 = await snowflake_tool._run(query=query)
    first_duration = asyncio.get_event_loop().time() - start_time

    # Second execution (should be cached)
    start_time = asyncio.get_event_loop().time()
    results2 = await snowflake_tool._run(query=query)
    second_duration = asyncio.get_event_loop().time() - start_time

    # Verify results
    assert results1 == results2
    assert len(results1) == 5
    assert second_duration < first_duration

    # Verify cache invalidation with different query
    different_query = "SELECT * FROM menu LIMIT 10"
    different_results = await snowflake_tool._run(query=different_query)
    assert len(different_results) == 10
    assert different_results != results1

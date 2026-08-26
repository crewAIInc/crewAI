import pytest

from crewai_tools.tools.databricks_query_tool.databricks_query_tool import (
    DatabricksQueryToolSchema,
)


def test_appends_default_limit_to_plain_select() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT * FROM orders")
    assert schema.query.rstrip(";").endswith("LIMIT 1000")


def test_appends_limit_when_table_name_contains_limit() -> None:
    """Regression: substring 'limit' in `limited_orders` used to skip the cap."""
    schema = DatabricksQueryToolSchema(query="SELECT * FROM limited_orders")
    assert schema.query.rstrip(";").upper().endswith("LIMIT 1000")


def test_appends_limit_when_column_is_named_limit() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT limit FROM orders")
    assert schema.query.rstrip(";").upper().endswith("LIMIT 1000")


def test_does_not_double_existing_limit() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT * FROM orders LIMIT 5")
    assert schema.query.rstrip(";") == "SELECT * FROM orders LIMIT 5"


def test_does_not_double_limit_all() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT * FROM orders LIMIT ALL")
    assert schema.query.rstrip(";") == "SELECT * FROM orders LIMIT ALL"


def test_does_not_double_fetch_first() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT * FROM orders FETCH FIRST 10 ROWS ONLY")
    assert "LIMIT" not in schema.query.upper()


def test_respects_custom_row_limit() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT * FROM limited_orders", row_limit=25)
    assert schema.query.rstrip(";").endswith("LIMIT 25")


def test_skips_append_when_row_limit_is_zero() -> None:
    schema = DatabricksQueryToolSchema(query="SELECT * FROM orders", row_limit=0)
    assert "LIMIT" not in schema.query.upper()


def test_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="Query cannot be empty"):
        DatabricksQueryToolSchema(query="   ")

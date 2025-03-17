import pytest


def test_databricks_sdk_import():
    """Test that databricks-sdk can be imported without errors.
    
    This test verifies that the databricks-sdk dependency is properly installed
    when using the tools extra, which is required by the databricks_query_tool.
    """
    try:
        import databricks.sdk
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import databricks.sdk: {e}")

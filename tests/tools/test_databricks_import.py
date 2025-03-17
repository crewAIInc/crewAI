import pytest
from importlib.metadata import version, PackageNotFoundError


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


def test_databricks_sdk_version():
    """Test that the installed databricks-sdk version meets requirements.
    
    The databricks-sdk should be version 0.46.0 or higher, but less than 1.0.0
    to maintain compatibility with the databricks_query_tool.
    """
    try:
        sdk_version = version("databricks-sdk")
        assert sdk_version >= "0.46.0", f"Databricks SDK version {sdk_version} is too old (< 0.46.0)"
        assert sdk_version < "1.0.0", f"Databricks SDK version {sdk_version} is too new (>= 1.0.0)"
    except PackageNotFoundError:
        pytest.fail("databricks-sdk package not found")


def test_databricks_core_functionality():
    """Test core functionality of the Databricks SDK.
    
    This test verifies that the WorkspaceClient class from the Databricks SDK
    has the expected attributes for SQL functionality.
    """
    from databricks.sdk import WorkspaceClient
    
    # Verify that the WorkspaceClient class has the expected attributes for SQL operations
    assert hasattr(WorkspaceClient, 'statement_execution'), "SQL statement execution functionality not available in WorkspaceClient"

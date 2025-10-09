from collections.abc import Generator
import os

from crewai_tools import SingleStoreSearchTool
from crewai_tools.tools.singlestore_search_tool import SingleStoreSearchToolSchema
import pytest
from singlestoredb import connect
from singlestoredb.server import docker


@pytest.fixture(scope="session")
def docker_server_url() -> Generator[str, None, None]:
    """Start a SingleStore Docker server for tests."""
    try:
        sdb = docker.start(license="")
        conn = sdb.connect()
        curr = conn.cursor()
        curr.execute("CREATE DATABASE test_crewai")
        curr.close()
        conn.close()
        yield sdb.connection_url
        sdb.stop()
    except Exception as e:
        pytest.skip(f"Could not start SingleStore Docker container: {e}")


@pytest.fixture(scope="function")
def clean_db_url(docker_server_url) -> Generator[str, None, None]:
    """Provide a clean database URL and clean up tables after test."""
    yield docker_server_url
    try:
        conn = connect(host=docker_server_url, database="test_crewai")
        curr = conn.cursor()
        curr.execute("SHOW TABLES")
        results = curr.fetchall()
        for result in results:
            curr.execute(f"DROP TABLE {result[0]}")
        curr.close()
        conn.close()
    except Exception:
        # Ignore cleanup errors
        pass


@pytest.fixture
def sample_table_setup(clean_db_url):
    """Set up sample tables for testing."""
    conn = connect(host=clean_db_url, database="test_crewai")
    curr = conn.cursor()

    # Create sample tables
    curr.execute(
        """
        CREATE TABLE employees (
            id INT PRIMARY KEY,
            name VARCHAR(100),
            department VARCHAR(50),
            salary DECIMAL(10,2)
        )
    """
    )

    curr.execute(
        """
        CREATE TABLE departments (
            id INT PRIMARY KEY,
            name VARCHAR(100),
            budget DECIMAL(12,2)
        )
    """
    )

    # Insert sample data
    curr.execute(
        """
        INSERT INTO employees VALUES
        (1, 'Alice Smith', 'Engineering', 75000.00),
        (2, 'Bob Johnson', 'Marketing', 65000.00),
        (3, 'Carol Davis', 'Engineering', 80000.00)
    """
    )

    curr.execute(
        """
        INSERT INTO departments VALUES
        (1, 'Engineering', 500000.00),
        (2, 'Marketing', 300000.00)
    """
    )

    curr.close()
    conn.close()
    return clean_db_url


class TestSingleStoreSearchTool:
    """Test suite for SingleStoreSearchTool."""

    def test_tool_creation_with_connection_params(self, sample_table_setup):
        """Test tool creation with individual connection parameters."""
        # Parse URL components for individual parameters
        url_parts = sample_table_setup.split("@")[1].split(":")
        host = url_parts[0]
        port = int(url_parts[1].split("/")[0])
        user = "root"
        password = sample_table_setup.split("@")[0].split(":")[2]
        tool = SingleStoreSearchTool(
            tables=[],
            host=host,
            port=port,
            user=user,
            password=password,
            database="test_crewai",
        )

        assert tool.name == "Search a database's table(s) content"
        assert "SingleStore" in tool.description
        assert (
            "employees(id int(11), name varchar(100), department varchar(50), salary decimal(10,2))"
            in tool.description.lower()
        )
        assert (
            "departments(id int(11), name varchar(100), budget decimal(12,2))"
            in tool.description.lower()
        )
        assert tool.args_schema == SingleStoreSearchToolSchema
        assert tool.connection_pool is not None

    def test_tool_creation_with_connection_url(self, sample_table_setup):
        """Test tool creation with connection URL."""
        tool = SingleStoreSearchTool(host=f"{sample_table_setup}/test_crewai")

        assert tool.name == "Search a database's table(s) content"
        assert tool.connection_pool is not None

    def test_tool_creation_with_specific_tables(self, sample_table_setup):
        """Test tool creation with specific table list."""
        tool = SingleStoreSearchTool(
            tables=["employees"],
            host=sample_table_setup,
            database="test_crewai",
        )

        # Check that description includes specific tables
        assert "employees" in tool.description
        assert "departments" not in tool.description

    def test_tool_creation_with_nonexistent_table(self, sample_table_setup):
        """Test tool creation fails with non-existent table."""

        with pytest.raises(ValueError, match="Table nonexistent does not exist"):
            SingleStoreSearchTool(
                tables=["employees", "nonexistent"],
                host=sample_table_setup,
                database="test_crewai",
            )

    def test_tool_creation_with_empty_database(self, clean_db_url):
        """Test tool creation fails with empty database."""

        with pytest.raises(ValueError, match="No tables found in the database"):
            SingleStoreSearchTool(host=clean_db_url, database="test_crewai")

    def test_description_generation(self, sample_table_setup):
        """Test that tool description is properly generated with table info."""

        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        # Check description contains table definitions
        assert "employees(" in tool.description
        assert "departments(" in tool.description
        assert "id int" in tool.description.lower()
        assert "name varchar" in tool.description.lower()

    def test_query_validation_select_allowed(self, sample_table_setup):
        """Test that SELECT queries are allowed."""
        os.environ["SINGLESTOREDB_URL"] = sample_table_setup
        tool = SingleStoreSearchTool(database="test_crewai")

        valid, message = tool._validate_query("SELECT * FROM employees")
        assert valid is True
        assert message == "Valid query"

    def test_query_validation_show_allowed(self, sample_table_setup):
        """Test that SHOW queries are allowed."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        valid, message = tool._validate_query("SHOW TABLES")
        assert valid is True
        assert message == "Valid query"

    def test_query_validation_case_insensitive(self, sample_table_setup):
        """Test that query validation is case insensitive."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        valid, _ = tool._validate_query("select * from employees")
        assert valid is True

        valid, _ = tool._validate_query("SHOW tables")
        assert valid is True

    def test_query_validation_insert_denied(self, sample_table_setup):
        """Test that INSERT queries are denied."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        valid, message = tool._validate_query(
            "INSERT INTO employees VALUES (4, 'Test', 'Test', 1000)"
        )
        assert valid is False
        assert "Only SELECT and SHOW queries are supported" in message

    def test_query_validation_update_denied(self, sample_table_setup):
        """Test that UPDATE queries are denied."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        valid, message = tool._validate_query("UPDATE employees SET salary = 90000")
        assert valid is False
        assert "Only SELECT and SHOW queries are supported" in message

    def test_query_validation_delete_denied(self, sample_table_setup):
        """Test that DELETE queries are denied."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        valid, message = tool._validate_query("DELETE FROM employees WHERE id = 1")
        assert valid is False
        assert "Only SELECT and SHOW queries are supported" in message

    def test_query_validation_non_string(self, sample_table_setup):
        """Test that non-string queries are rejected."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        valid, message = tool._validate_query(123)
        assert valid is False
        assert "Search query must be a string" in message

    def test_run_select_query(self, sample_table_setup):
        """Test executing a SELECT query."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        result = tool._run("SELECT * FROM employees ORDER BY id")

        assert "Search Results:" in result
        assert "Alice Smith" in result
        assert "Bob Johnson" in result
        assert "Carol Davis" in result

    def test_run_filtered_query(self, sample_table_setup):
        """Test executing a filtered SELECT query."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        result = tool._run(
            "SELECT name FROM employees WHERE department = 'Engineering'"
        )

        assert "Search Results:" in result
        assert "Alice Smith" in result
        assert "Carol Davis" in result
        assert "Bob Johnson" not in result

    def test_run_show_query(self, sample_table_setup):
        """Test executing a SHOW query."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        result = tool._run("SHOW TABLES")

        assert "Search Results:" in result
        assert "employees" in result
        assert "departments" in result

    def test_run_empty_result(self, sample_table_setup):
        """Test executing a query that returns no results."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        result = tool._run("SELECT * FROM employees WHERE department = 'NonExistent'")

        assert result == "No results found."

    def test_run_invalid_query_syntax(self, sample_table_setup):
        """Test executing a query with invalid syntax."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        result = tool._run("SELECT * FORM employees")  # Intentional typo

        assert "Error executing search query:" in result

    def test_run_denied_query(self, sample_table_setup):
        """Test that denied queries return appropriate error message."""
        tool = SingleStoreSearchTool(host=sample_table_setup, database="test_crewai")

        result = tool._run("DELETE FROM employees")

        assert "Invalid search query:" in result
        assert "Only SELECT and SHOW queries are supported" in result

    def test_connection_pool_usage(self, sample_table_setup):
        """Test that connection pooling works correctly."""
        tool = SingleStoreSearchTool(
            host=sample_table_setup,
            database="test_crewai",
            pool_size=2,
        )

        # Execute multiple queries to test pool usage
        results = []
        for _ in range(5):
            result = tool._run("SELECT COUNT(*) FROM employees")
            results.append(result)

        # All queries should succeed
        for result in results:
            assert "Search Results:" in result
            assert "3" in result  # Count of employees

    def test_tool_schema_validation(self):
        """Test that the tool schema validation works correctly."""
        # Valid input
        valid_input = SingleStoreSearchToolSchema(search_query="SELECT * FROM test")
        assert valid_input.search_query == "SELECT * FROM test"

        # Test that description is present
        schema_dict = SingleStoreSearchToolSchema.model_json_schema()
        assert "search_query" in schema_dict["properties"]
        assert "description" in schema_dict["properties"]["search_query"]

    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        with pytest.raises(Exception):
            # This should fail due to invalid connection parameters
            SingleStoreSearchTool(
                host="invalid_host",
                port=9999,
                user="invalid_user",
                password="invalid_password",
                database="invalid_db",
            )

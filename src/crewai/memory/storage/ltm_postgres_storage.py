import json
from typing import Any, Dict, List, Optional, Union

import psycopg
from psycopg.rows import dict_row

try:
    # First try importing from psycopg.pool
    from psycopg.pool import ConnectionPool
    HAS_CONNECTION_POOL = True
except ImportError:
    try:
        # Then try from psycopg_pool as a separate module
        from psycopg_pool import ConnectionPool
        HAS_CONNECTION_POOL = True
    except ImportError:
        # For older versions of psycopg or when pool extras aren't installed
        HAS_CONNECTION_POOL = False
        ConnectionPool = None

from crewai.utilities import Printer
from crewai.utilities.postgres_config import (
    sanitize_connection_string,
    escape_like,
    validate_identifier,
    safe_parse_json,
)


class PostgresStorageError(Exception):
    """Base exception for PostgreSQL storage errors."""

    pass


class DatabaseConnectionError(PostgresStorageError):
    """Exception raised when there's a problem connecting to the database."""

    pass


class DatabaseQueryError(PostgresStorageError):
    """Exception raised when there's an error executing a database query."""

    pass


class DatabaseConfigurationError(PostgresStorageError):
    """Exception raised when there's an issue with the database configuration."""

    pass


class DatabaseSecurityError(PostgresStorageError):
    """Exception raised when there's a security-related issue with the database operation."""

    pass


class LTMPostgresStorage:
    """
    Postgres storage implementation for Long Term Memory data storage.
    Compatible with Postgres 16 and later versions.

    Supports connection pooling for improved performance in high-volume environments.
    Includes security features like connection string sanitization, schema/table name
    validation, and parameterized queries to prevent SQL injection.
    """

    def __init__(
        self,
        connection_string: str,
        schema: str = "public",
        table_name: str = "long_term_memories",
        min_pool_size: int = 1,
        max_pool_size: int = 5,
        use_connection_pool: bool = True,
    ) -> None:
        """
        Initialize the Postgres storage.

        Args:
            connection_string: Postgres connection string (e.g., "postgresql://user:password@localhost:5432/dbname")
            schema: Database schema (defaults to "public")
            table_name: Table name for storing memories (defaults to "long_term_memories")
            min_pool_size: Minimum size of the connection pool (default 1)
            max_pool_size: Maximum size of the connection pool (default 5)
            use_connection_pool: Whether to use connection pooling (default True)

        Raises:
            DatabaseConfigurationError: If the connection string, schema, or table name is invalid
            DatabaseConnectionError: If there's an issue connecting to the database
        """
        self.connection_string = connection_string
        # Store sanitized version for logging
        self._safe_conn_string = sanitize_connection_string(connection_string)
        self._printer: Printer = Printer()

        # Validate schema name
        is_valid, error_message = validate_identifier(schema, "schema")
        if not is_valid:
            self._printer.print(
                content=f"MEMORY ERROR: {error_message}",
                color="red",
            )
            raise DatabaseConfigurationError(error_message)

        # Validate table name
        is_valid, error_message = validate_identifier(table_name, "table")
        if not is_valid:
            self._printer.print(
                content=f"MEMORY ERROR: {error_message}",
                color="red",
            )
            raise DatabaseConfigurationError(error_message)

        self.schema = schema
        self.table_name = table_name
        self.full_table_name = f"{self.schema}.{self.table_name}"

        # Only use connection pooling if explicitly requested AND available
        self.use_connection_pool = use_connection_pool and HAS_CONNECTION_POOL

        if not HAS_CONNECTION_POOL and use_connection_pool:
            self._printer.print(
                content="Connection pooling requested but psycopg pool module not available. "
                "Install with 'pip install \"psycopg[pool]\"'. "
                "Falling back to direct connections.",
                color="yellow",
            )

        # Create connection pool if enabled and available
        if self.use_connection_pool:
            try:
                self.pool = ConnectionPool(
                    self.connection_string,
                    min_size=min_pool_size,
                    max_size=max_pool_size,
                    # Configure pool behavior
                    kwargs={"row_factory": dict_row},
                )
            except Exception as e:
                self._printer.print(
                    content=f"Failed to create connection pool: {e}. "
                    f"Connection string: {self._safe_conn_string}. "
                    f"Falling back to direct connections.",
                    color="red",
                )
                self.pool = None
                self.use_connection_pool = False
        else:
            self.pool = None

        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the Postgres database and creates LTM table if it doesn't exist.

        Raises:
            DatabaseConnectionError: If there's an issue connecting to the database
            DatabaseQueryError: If there's an error executing the initialization queries
            PostgresStorageError: For any other unexpected errors
        """
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._create_schema_and_table(conn)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._create_schema_and_table(conn)

        except psycopg.OperationalError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database connection failed during initialization: {e}. "
                f"Connection string: {self._safe_conn_string}",
                color="red",
            )
            raise DatabaseConnectionError(
                "Failed to connect to PostgreSQL during initialization"
            )
        except psycopg.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database error during initialization: {e}",
                color="red",
            )
            raise DatabaseQueryError("Error executing query during initialization")
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An unexpected error occurred during database initialization: {e}",
                color="red",
            )
            raise PostgresStorageError("Unexpected error during initialization")

    def _create_schema_and_table(self, conn):
        """
        Create schema and table with the given connection.

        Args:
            conn: The database connection
        """
        # SECURITY NOTE: schema and table names are validated in __init__ to prevent SQL injection
        with conn.cursor() as cursor:
            # Create schema if it doesn't exist
            create_schema_sql = "CREATE SCHEMA IF NOT EXISTS " + self.schema  # nosec B608
            cursor.execute(create_schema_sql)

            # Create the table if it doesn't exist
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS {0} (
                id SERIAL PRIMARY KEY,
                task_description TEXT,
                metadata JSONB,
                datetime TEXT,
                score REAL
            )
            """.format(self.full_table_name)  # nosec B608
            cursor.execute(create_table_sql)

            # Create index on task_description for faster lookups
            create_index_sql = """
            CREATE INDEX IF NOT EXISTS idx_{0}_task_description 
            ON {1} (task_description)
            """.format(self.table_name, self.full_table_name)  # nosec B608
            cursor.execute(create_index_sql)

            conn.commit()

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        datetime: str,
        score: Union[int, float],
    ) -> None:
        """
        Saves data to the LTM table with error handling.

        Args:
            task_description: The description of the task
            metadata: Dictionary containing metadata about the memory
            datetime: ISO formatted datetime string
            score: Numerical score for the memory

        Raises:
            DatabaseConnectionError: If there's an issue connecting to the database
            DatabaseQueryError: If there's an error executing the save query
            PostgresStorageError: For any other unexpected errors
        """
        # Validate inputs
        if not task_description:
            raise ValueError("Task description cannot be empty")

        if not isinstance(metadata, dict):
            self._printer.print(
                content="WARNING: Metadata is not a dictionary. Attempting to convert.",
                color="yellow",
            )
            metadata = safe_parse_json(metadata)

        if not isinstance(score, (int, float)):
            raise ValueError(f"Score must be a number, got {type(score).__name__}")

        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._perform_save(
                        conn, task_description, metadata, datetime, score
                    )
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._perform_save(
                        conn, task_description, metadata, datetime, score
                    )

        except psycopg.OperationalError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database connection failed while saving to LTM: {e}. "
                f"Connection string: {self._safe_conn_string}",
                color="red",
            )
            raise DatabaseConnectionError(
                "Failed to connect to PostgreSQL while saving"
            )
        except psycopg.DataError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Invalid data format while saving to LTM: {e}. "
                f"Task: {task_description[:30]}{'...' if len(task_description) > 30 else ''}, "
                f"Score: {score}",
                color="red",
            )
            raise DatabaseQueryError("Invalid data format while saving")
        except psycopg.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database error while saving to LTM: {e}",
                color="red",
            )
            raise DatabaseQueryError("Error executing query while saving")
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An unexpected error occurred while saving to LTM: {e}",
                color="red",
            )
            raise PostgresStorageError("Unexpected error while saving")

    def _perform_save(self, conn, task_description, metadata, datetime, score):
        """
        Execute the save operation with the given connection.

        Args:
            conn: The database connection
            task_description: The description of the task
            metadata: Dictionary containing metadata about the memory
            datetime: ISO formatted datetime string
            score: Numerical score for the memory
        """
        with conn.cursor() as cursor:
            # Safely convert metadata to JSON
            try:
                metadata_json = json.dumps(metadata)
            except (TypeError, OverflowError, ValueError) as e:
                self._printer.print(
                    content=f"WARNING: Error converting metadata to JSON: {e}. Using empty dict.",
                    color="yellow",
                )
                metadata_json = "{}"

            # SECURITY NOTE: table name is validated in __init__ to prevent SQL injection
            insert_sql = """
            INSERT INTO {0} 
            (task_description, metadata, datetime, score)
            VALUES (%s, %s, %s, %s)
            """.format(self.full_table_name)  # nosec B608
            cursor.execute(
                insert_sql,
                (task_description, metadata_json, datetime, score),
            )
            conn.commit()

    def save_many(
        self,
        items: List[Dict[str, Any]],
    ) -> None:
        """
        Saves multiple data items to the LTM table in a single transaction.

        Args:
            items: List of dictionaries with keys: task_description, metadata, datetime, score

        Raises:
            DatabaseConnectionError: If there's an issue connecting to the database
            DatabaseQueryError: If there's an error executing the batch save query
            PostgresStorageError: For any other unexpected errors
        """
        if not items:
            return

        # Validate items list
        for i, item in enumerate(items):
            required_keys = ["task_description", "metadata", "datetime", "score"]
            missing_keys = [key for key in required_keys if key not in item]
            if missing_keys:
                raise ValueError(
                    f"Item at index {i} is missing required keys: {', '.join(missing_keys)}"
                )

            if not isinstance(item["score"], (int, float)):
                raise ValueError(
                    f"Score for item at index {i} must be a number, got {type(item['score']).__name__}"
                )

        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._perform_save_many(conn, items)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._perform_save_many(conn, items)

        except psycopg.OperationalError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database connection failed while batch saving to LTM: {e}. "
                f"Connection string: {self._safe_conn_string}",
                color="red",
            )
            raise DatabaseConnectionError(
                "Failed to connect to PostgreSQL during batch save"
            )
        except psycopg.DataError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Invalid data format while batch saving to LTM: {e}. "
                f"Items count: {len(items)}",
                color="red",
            )
            raise DatabaseQueryError("Invalid data format during batch save")
        except psycopg.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database error while batch saving to LTM: {e}",
                color="red",
            )
            raise DatabaseQueryError("Error executing query during batch save")
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An unexpected error occurred while batch saving to LTM: {e}",
                color="red",
            )
            raise PostgresStorageError("Unexpected error during batch save")

    def _perform_save_many(self, conn, items):
        """
        Execute the batch save operation with the given connection.

        Args:
            conn: The database connection
            items: List of dictionaries with keys: task_description, metadata, datetime, score
        """
        with conn.cursor() as cursor:
            values = []
            for item in items:
                # Safely convert metadata to JSON
                try:
                    metadata_json = json.dumps(item["metadata"])
                except (TypeError, OverflowError) as e:
                    self._printer.print(
                        content=f"WARNING: Error converting metadata to JSON: {e}. Using empty dict.",
                        color="yellow",
                    )
                    metadata_json = "{}"

                values.append(
                    (
                        item["task_description"],
                        metadata_json,
                        item["datetime"],
                        item["score"],
                    )
                )

            # SECURITY NOTE: table name is validated in __init__ to prevent SQL injection
            insert_many_sql = """
            INSERT INTO {0} 
            (task_description, metadata, datetime, score)
            VALUES (%s, %s, %s, %s)
            """.format(self.full_table_name)  # nosec B608
            cursor.executemany(
                insert_many_sql,
                values,
            )
            conn.commit()

    def load(
        self, task_description: str, latest_n: int
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Queries the LTM table by task description with error handling.

        Args:
            task_description: The description of the task to query for
            latest_n: Maximum number of results to return

        Returns:
            List of memory items or None if not found/error

        Raises:
            DatabaseConnectionError: If there's an issue connecting to the database
            DatabaseQueryError: If there's an error executing the query
            PostgresStorageError: For any other unexpected errors
        """
        # Validate inputs
        if not task_description:
            raise ValueError("Task description cannot be empty")

        if not isinstance(latest_n, int) or latest_n < 1:
            raise ValueError(f"latest_n must be a positive integer, got {latest_n}")

        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    return self._perform_load(conn, task_description, latest_n)
            else:
                with psycopg.connect(
                    self.connection_string, row_factory=dict_row
                ) as conn:
                    return self._perform_load(conn, task_description, latest_n)

        except psycopg.OperationalError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database connection failed while querying LTM: {e}. "
                f"Connection string: {self._safe_conn_string}",
                color="red",
            )
            raise DatabaseConnectionError(
                "Failed to connect to PostgreSQL during query"
            )
        except psycopg.DataError as e:
            # Escape task description for safe logging
            safe_task_description = escape_like(task_description)
            self._printer.print(
                content=f"MEMORY ERROR: Invalid data format while querying LTM: {e}. "
                f"Task description: {safe_task_description[:50]}{'...' if len(safe_task_description) > 50 else ''}, "
                f"Limit: {latest_n}",
                color="red",
            )
            raise DatabaseQueryError("Invalid data format during query")
        except psycopg.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database error while querying LTM: {e}",
                color="red",
            )
            raise DatabaseQueryError("Error executing query")
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An unexpected error occurred while querying LTM: {e}",
                color="red",
            )
            raise PostgresStorageError("Unexpected error during query")

    def _perform_load(self, conn, task_description, latest_n):
        """
        Execute the load operation with the given connection.

        Args:
            conn: The database connection
            task_description: The description of the task to query for
            latest_n: Maximum number of results to return

        Returns:
            List of memory items or None if not found
        """
        with conn.cursor() as cursor:
            # SECURITY NOTE: table name is validated in __init__ to prevent SQL injection
            # Use parameterized query to prevent SQL injection
            select_sql = """
            SELECT metadata, datetime, score
            FROM {0}
            WHERE task_description = %s
            ORDER BY datetime DESC, score ASC
            LIMIT %s
            """.format(self.full_table_name)  # nosec B608
            cursor.execute(
                select_sql,
                (task_description, latest_n),
            )
            rows = cursor.fetchall()

            if rows:
                return [
                    {
                        # Handle potential JSON parsing errors gracefully
                        "metadata": safe_parse_json(row["metadata"]),
                        "datetime": row["datetime"],
                        "score": row["score"],
                    }
                    for row in rows
                ]
        return None

    def reset(
        self,
    ) -> None:
        """
        Resets the LTM table by deleting all rows (with error handling).

        Raises:
            DatabaseConnectionError: If there's an issue connecting to the database
            DatabaseQueryError: If there's an error executing the reset query
            PostgresStorageError: For any other unexpected errors
        """
        try:
            # Use the connection pool if available, otherwise create a new connection
            if self.use_connection_pool:
                with self.pool.connection() as conn:
                    self._perform_reset(conn)
            else:
                with psycopg.connect(self.connection_string) as conn:
                    self._perform_reset(conn)

        except psycopg.OperationalError as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database connection failed while resetting LTM: {e}. "
                f"Connection string: {self._safe_conn_string}",
                color="red",
            )
            raise DatabaseConnectionError(
                "Failed to connect to PostgreSQL during reset"
            )
        except psycopg.Error as e:
            self._printer.print(
                content=f"MEMORY ERROR: Database error while resetting LTM: {e}",
                color="red",
            )
            raise DatabaseQueryError("Error executing reset query")
        except Exception as e:
            self._printer.print(
                content=f"MEMORY ERROR: An unexpected error occurred while resetting LTM: {e}",
                color="red",
            )
            raise PostgresStorageError("Unexpected error during reset")

    def _perform_reset(self, conn):
        """
        Execute the reset operation with the given connection.

        Args:
            conn: The database connection
        """
        # SECURITY NOTE: table name is validated in __init__ to prevent SQL injection
        with conn.cursor() as cursor:
            delete_sql = "DELETE FROM {0}".format(self.full_table_name)  # nosec B608
            cursor.execute(delete_sql)
            conn.commit()

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            self: The storage instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager, ensuring resources are properly closed.

        Args:
            exc_type: Exception type if an exception was raised in the context
            exc_val: Exception value if an exception was raised in the context
            exc_tb: Exception traceback if an exception was raised in the context
        """
        self.cleanup()

    def cleanup(self):
        """
        Clean up resources and connections.

        This method ensures proper cleanup of database resources,
        even if an error occurs during the closing process.
        """
        if self.pool:
            try:
                self.pool.close()
            except Exception as e:
                self._printer.print(
                    content=f"WARNING: Error closing connection pool: {e}",
                    color="yellow",
                )

    # Keep close() for backward compatibility
    def close(self):
        """
        Close connection pool and release resources.

        This method ensures proper cleanup of database resources,
        even if an error occurs during the closing process.

        Note: This method is an alias for cleanup() and is maintained for backward compatibility.
        """
        self.cleanup()

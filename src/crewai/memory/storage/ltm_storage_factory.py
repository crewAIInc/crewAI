from typing import Optional, Union, Tuple

from crewai.memory.storage.ltm_postgres_storage import (
    LTMPostgresStorage,
    DatabaseConfigurationError,
)
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.utilities.postgres_config import (
    get_postgres_config,
    get_postgres_connection_string,
    validate_identifier,
)


class LTMStorageFactory:
    """
    Factory class for creating LTM storage instances based on configuration.

    This factory provides a unified way to create various storage backends
    for long-term memory, making it easy to switch between SQLite and Postgres
    or potentially other backends in the future.

    Features include:
    - Configuration validation before creating storage instances
    - Automatic environment variable detection
    - Support for connection pooling with Postgres
    - Error handling with detailed messages
    """

    @staticmethod
    def _validate_postgres_config(
        connection_string: Optional[str],
        min_pool_size: Optional[int],
        max_pool_size: Optional[int],
        schema: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Validate PostgreSQL configuration parameters.

        Args:
            connection_string: PostgreSQL connection string
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            schema: Database schema name
            table_name: Database table name

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Validate connection string format if provided
        if connection_string:
            if not connection_string.startswith("postgresql://"):
                return False, (
                    "Invalid PostgreSQL connection string format. "
                    "Must start with 'postgresql://'."
                )

            # Check for potentially dangerous connection string patterns
            dangerous_patterns = [";", "--", "/*", "*/", "DROP ", "DELETE ", "UPDATE "]
            for pattern in dangerous_patterns:
                if pattern.lower() in connection_string.lower():
                    return (
                        False,
                        f"Connection string contains potentially unsafe pattern: {pattern}",
                    )

        # Validate pool size configuration
        if min_pool_size is not None and max_pool_size is not None:
            if min_pool_size < 1:
                return False, "min_pool_size must be at least 1"

            if max_pool_size < 1:
                return False, "max_pool_size must be at least 1"

            if min_pool_size > max_pool_size:
                return False, "min_pool_size cannot be greater than max_pool_size"

        # Validate schema name if provided
        if schema is not None:
            is_valid, error_message = validate_identifier(schema, "schema")
            if not is_valid:
                return False, error_message

        # Validate table name if provided
        if table_name is not None:
            is_valid, error_message = validate_identifier(table_name, "table")
            if not is_valid:
                return False, error_message

        return True, ""

    @staticmethod
    def create_storage(
        storage_type: str = "sqlite",
        path: Optional[str] = None,
        connection_string: Optional[str] = None,
        schema: Optional[str] = None,
        table_name: Optional[str] = None,
        min_pool_size: Optional[int] = None,
        max_pool_size: Optional[int] = None,
        use_connection_pool: Optional[bool] = None,
    ) -> Union[LTMSQLiteStorage, LTMPostgresStorage]:
        """
        Create and return a storage instance based on the specified type.

        Args:
            storage_type: Type of storage to create ("sqlite" or "postgres")
            path: File path for SQLite database (only used with SQLite)
            connection_string: Connection string for Postgres (only used with Postgres)
            schema: Database schema for Postgres (only used with Postgres)
            table_name: Table name for Postgres (only used with Postgres)
            min_pool_size: Minimum connection pool size (only used with Postgres)
            max_pool_size: Maximum connection pool size (only used with Postgres)
            use_connection_pool: Whether to use connection pooling (only used with Postgres)

        Returns:
            Storage instance of the requested type

        Raises:
            ValueError: If an unsupported storage type is specified
            DatabaseConfigurationError: If required parameters are missing or invalid
        """
        if storage_type.lower() == "sqlite":
            # Validate SQLite path if provided
            if path is not None and not path:
                raise DatabaseConfigurationError(
                    "SQLite database path cannot be an empty string"
                )

            return LTMSQLiteStorage(db_path=path)
        elif storage_type.lower() == "postgres":
            # Get configuration from environment if not provided
            config = get_postgres_config()

            # Use provided values or fall back to environment config
            conn_string = connection_string or get_postgres_connection_string()
            if not conn_string:
                raise DatabaseConfigurationError(
                    "A connection string must be provided for Postgres storage, "
                    "either directly or via CREWAI_PG_* environment variables"
                )

            pg_schema = schema or config["schema"]
            pg_table = table_name or config["table"]
            pg_min_pool = (
                min_pool_size if min_pool_size is not None else config["min_pool"]
            )
            pg_max_pool = (
                max_pool_size if max_pool_size is not None else config["max_pool"]
            )
            pg_use_pool = (
                use_connection_pool
                if use_connection_pool is not None
                else config["enable_pool"]
            )

            # Validate PostgreSQL configuration
            is_valid, error_message = LTMStorageFactory._validate_postgres_config(
                connection_string=conn_string,
                min_pool_size=pg_min_pool,
                max_pool_size=pg_max_pool,
                schema=pg_schema,
                table_name=pg_table,
            )

            if not is_valid:
                raise DatabaseConfigurationError(error_message)

            return LTMPostgresStorage(
                connection_string=conn_string,
                schema=pg_schema,
                table_name=pg_table,
                min_pool_size=pg_min_pool,
                max_pool_size=pg_max_pool,
                use_connection_pool=pg_use_pool,
            )
        else:
            raise ValueError(
                f"Unsupported storage type: {storage_type}. "
                "Supported types are 'sqlite' and 'postgres'."
            )

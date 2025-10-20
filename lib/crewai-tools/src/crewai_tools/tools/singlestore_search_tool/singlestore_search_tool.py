from collections.abc import Callable
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


try:
    from singlestoredb import connect
    from sqlalchemy.pool import QueuePool

    SINGLSTORE_AVAILABLE = True

except ImportError:
    SINGLSTORE_AVAILABLE = False


class SingleStoreSearchToolSchema(BaseModel):
    """Input schema for SingleStoreSearchTool.

    This schema defines the expected input format for the search tool,
    ensuring that only valid SELECT and SHOW queries are accepted.
    """

    search_query: str = Field(
        ...,
        description=(
            "Mandatory semantic search query you want to use to search the database's content. "
            "Only SELECT and SHOW queries are supported."
        ),
    )


class SingleStoreSearchTool(BaseTool):
    """A tool for performing semantic searches on SingleStore database tables.

    This tool provides a safe interface for executing SELECT and SHOW queries
    against a SingleStore database with connection pooling for optimal performance.
    """

    name: str = "Search a database's table(s) content"
    description: str = (
        "A tool that can be used to semantic search a query from a database."
    )
    args_schema: type[BaseModel] = SingleStoreSearchToolSchema

    package_dependencies: list[str] = Field(
        default_factory=lambda: ["singlestoredb", "SQLAlchemy"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SINGLESTOREDB_URL",
                description="A comprehensive URL string that can encapsulate host, port,"
                " username, password, and database information, often used in environments"
                " like SingleStore notebooks or specific frameworks."
                " For example: 'me:p455w0rd@s2-host.com/my_db'",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_HOST",
                description="Specifies the hostname, IP address, or URL of"
                " the SingleStoreDB workspace or cluster",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_PORT",
                description="Defines the port number on which the"
                " SingleStoreDB server is listening",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_USER",
                description="Specifies the database user name",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_PASSWORD",
                description="Specifies the database user password",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_DATABASE",
                description="Name of the database to connect to",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_SSL_KEY",
                description="File containing SSL key",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_SSL_CERT",
                description="File containing SSL certificate",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_SSL_CA",
                description="File containing SSL certificate authority",
                required=False,
                default=None,
            ),
            EnvVar(
                name="SINGLESTOREDB_CONNECT_TIMEOUT",
                description="The timeout for connecting to the database in seconds",
                required=False,
                default=None,
            ),
        ]
    )

    connection_args: dict = Field(default_factory=dict)
    connection_pool: Any | None = None

    def __init__(
        self,
        tables: list[str] | None = None,
        # Basic connection parameters
        host: str | None = None,
        user: str | None = None,
        password: str | None = None,
        port: int | None = None,
        database: str | None = None,
        driver: str | None = None,
        # Connection behavior options
        pure_python: bool | None = None,
        local_infile: bool | None = None,
        charset: str | None = None,
        # SSL/TLS configuration
        ssl_key: str | None = None,
        ssl_cert: str | None = None,
        ssl_ca: str | None = None,
        ssl_disabled: bool | None = None,
        ssl_cipher: str | None = None,
        ssl_verify_cert: bool | None = None,
        tls_sni_servername: str | None = None,
        ssl_verify_identity: bool | None = None,
        # Advanced connection options
        conv: dict[int, Callable[..., Any]] | None = None,
        credential_type: str | None = None,
        autocommit: bool | None = None,
        # Result formatting options
        results_type: str | None = None,
        buffered: bool | None = None,
        results_format: str | None = None,
        program_name: str | None = None,
        conn_attrs: dict[str, str] | None = None,
        # Query execution options
        multi_statements: bool | None = None,
        client_found_rows: bool | None = None,
        connect_timeout: int | None = None,
        # Data type handling
        nan_as_null: bool | None = None,
        inf_as_null: bool | None = None,
        encoding_errors: str | None = None,
        track_env: bool | None = None,
        enable_extended_data_types: bool | None = None,
        vector_data_format: str | None = None,
        parse_json: bool | None = None,
        # Connection pool configuration
        pool_size: int | None = 5,
        max_overflow: int | None = 10,
        timeout: float | None = 30,
        **kwargs,
    ):
        """Initialize the SingleStore search tool.

        Args:
            tables: List of table names to work with. If empty, all tables will be used.
            host: Database host address
            user: Database username
            password: Database password
            port: Database port number
            database: Database name
            pool_size: Maximum number of connections in the pool
            max_overflow: Maximum overflow connections beyond pool_size
            timeout: Connection timeout in seconds
            **kwargs: Additional arguments passed to the parent class
        """
        if conn_attrs is None:
            conn_attrs = {}
        if tables is None:
            tables = []
        if not SINGLSTORE_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'singlestore' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(
                        ["uv", "add", "crewai-tools[singlestore]"],  # noqa: S607
                        check=True,
                    )

                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install singlestore package") from e
            else:
                raise ImportError(
                    "`singlestore` package not found, please run `uv add crewai-tools[singlestore]`"
                )

        # Set the data type for the parent class
        kwargs["data_type"] = "singlestore"
        super().__init__(**kwargs)

        # Build connection arguments dictionary with sensible defaults
        self.connection_args = {
            # Basic connection parameters
            "host": host,
            "user": user,
            "password": password,
            "port": port,
            "database": database,
            "driver": driver,
            # Connection behavior
            "pure_python": pure_python,
            "local_infile": local_infile,
            "charset": charset,
            # SSL/TLS settings
            "ssl_key": ssl_key,
            "ssl_cert": ssl_cert,
            "ssl_ca": ssl_ca,
            "ssl_disabled": ssl_disabled,
            "ssl_cipher": ssl_cipher,
            "ssl_verify_cert": ssl_verify_cert,
            "tls_sni_servername": tls_sni_servername,
            "ssl_verify_identity": ssl_verify_identity,
            # Advanced options
            "conv": conv or {},
            "credential_type": credential_type,
            "autocommit": autocommit,
            # Result formatting
            "results_type": results_type,
            "buffered": buffered,
            "results_format": results_format,
            "program_name": program_name,
            "conn_attrs": conn_attrs or {},
            # Query execution
            "multi_statements": multi_statements,
            "client_found_rows": client_found_rows,
            "connect_timeout": connect_timeout or 10,  # Default: 10 seconds
            # Data type handling with defaults
            "nan_as_null": nan_as_null or False,
            "inf_as_null": inf_as_null or False,
            "encoding_errors": encoding_errors or "strict",
            "track_env": track_env or False,
            "enable_extended_data_types": enable_extended_data_types or False,
            "vector_data_format": vector_data_format or "binary",
            "parse_json": parse_json or True,
        }

        # Ensure connection attributes are properly initialized
        if "conn_attrs" not in self.connection_args or not self.connection_args.get(
            "conn_attrs"
        ):
            self.connection_args["conn_attrs"] = dict()

        # Add tool identification to connection attributes
        self.connection_args["conn_attrs"]["_connector_name"] = (
            "crewAI SingleStore Tool"
        )
        self.connection_args["conn_attrs"]["_connector_version"] = "1.0"

        # Initialize connection pool for efficient connection management
        self.connection_pool = QueuePool(
            creator=self._create_connection,  # type: ignore[arg-type]
            pool_size=pool_size or 5,
            max_overflow=max_overflow or 10,
            timeout=timeout or 30.0,
        )

        # Validate database schema and initialize table information
        self._initialize_tables(tables)

    def _initialize_tables(self, tables: list[str]) -> None:
        """Initialize and validate the tables that this tool will work with.

        Args:
            tables: List of table names to validate and use

        Raises:
            ValueError: If no tables exist or specified tables don't exist
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Get all existing tables in the database
                cursor.execute("SHOW TABLES")
                existing_tables = {table[0] for table in cursor.fetchall()}

                # Validate that the database has tables
                if not existing_tables or len(existing_tables) == 0:
                    raise ValueError(
                        "No tables found in the database. "
                        "Please ensure the database is initialized with the required tables."
                    )

                # Use all tables if none specified
                if not tables or len(tables) == 0:
                    tables = list(existing_tables)

                # Build table definitions for description
                table_definitions = []
                for table in tables:
                    if table not in existing_tables:
                        raise ValueError(
                            f"Table {table} does not exist in the database. "
                            f"Please ensure the table is created."
                        )

                    # Get column information for each table
                    cursor.execute(f"SHOW COLUMNS FROM {table}")
                    columns = cursor.fetchall()
                    column_info = ", ".join(f"{row[0]} {row[1]}" for row in columns)
                    table_definitions.append(f"{table}({column_info})")
        finally:
            # Ensure the connection is returned to the pool
            conn.close()

        # Update the tool description with actual table information
        self.description = (
            f"A tool that can be used to semantic search a query from a SingleStore "
            f"database's {', '.join(table_definitions)} table(s) content."
        )
        self._generate_description()

    def _get_connection(self) -> Any:
        """Get a connection from the connection pool.

        Returns:
            Connection: A SingleStore database connection

        Raises:
            Exception: If connection cannot be established
        """
        try:
            return self.connection_pool.connect()  # type: ignore[union-attr]
        except Exception:
            # Re-raise the exception to be handled by the caller
            raise

    def _create_connection(self) -> Any:
        """Create a new SingleStore connection.

        This method is used by the connection pool to create new connections
        when needed.

        Returns:
            Connection: A new SingleStore database connection

        Raises:
            Exception: If connection cannot be created
        """
        try:
            return connect(**self.connection_args)
        except Exception:
            # Re-raise the exception to be handled by the caller
            raise

    def _validate_query(self, search_query: str) -> tuple[bool, str]:
        """Validate the search query to ensure it's safe to execute.

        Only SELECT and SHOW statements are allowed for security reasons.

        Args:
            search_query: The SQL query to validate

        Returns:
            tuple: (is_valid: bool, message: str)
        """
        # Check if the input is a string
        if not isinstance(search_query, str):
            return False, "Search query must be a string."

        # Remove leading/trailing whitespace and convert to lowercase for checking
        query_lower = search_query.strip().lower()

        # Allow only SELECT and SHOW statements
        if not (query_lower.startswith(("select", "show"))):
            return (
                False,
                "Only SELECT and SHOW queries are supported for security reasons.",
            )

        return True, "Valid query"

    def _run(self, search_query: str) -> Any:
        """Execute the search query against the SingleStore database.

        Args:
            search_query: The SQL query to execute
            **kwargs: Additional keyword arguments (unused)

        Returns:
            str: Formatted search results or error message
        """
        # Validate the query before execution
        valid, message = self._validate_query(search_query)
        if not valid:
            return f"Invalid search query: {message}"

        # Execute the query using a connection from the pool
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                try:
                    # Execute the validated search query
                    cursor.execute(search_query)
                    results = cursor.fetchall()

                    # Handle empty results
                    if not results:
                        return "No results found."

                    # Format the results for readable output
                    formatted_results = "\n".join(
                        [", ".join([str(item) for item in row]) for row in results]
                    )
                    return f"Search Results:\n{formatted_results}"

                except Exception as e:
                    return f"Error executing search query: {e}"

        finally:
            # Ensure the connection is returned to the pool
            conn.close()

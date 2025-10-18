from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import TYPE_CHECKING, Any

from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, ConfigDict, Field, SecretStr


if TYPE_CHECKING:
    # Import types for type checking only
    from snowflake.connector.connection import (  # type: ignore[import-not-found]
        SnowflakeConnection,
    )
    from snowflake.connector.errors import (  # type: ignore[import-not-found]
        DatabaseError,
        OperationalError,
    )

try:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    import snowflake.connector  # type: ignore[import-not-found]

    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Cache for query results
_query_cache: dict[str, list[dict[str, Any]]] = {}


class SnowflakeConfig(BaseModel):
    """Configuration for Snowflake connection."""

    model_config = ConfigDict(protected_namespaces=())

    account: str = Field(
        ..., description="Snowflake account identifier", pattern=r"^[a-zA-Z0-9\-_]+$"
    )
    user: str = Field(..., description="Snowflake username")
    password: SecretStr | None = Field(None, description="Snowflake password")
    private_key_path: str | None = Field(None, description="Path to private key file")
    warehouse: str | None = Field(None, description="Snowflake warehouse")
    database: str | None = Field(None, description="Default database")
    snowflake_schema: str | None = Field(None, description="Default schema")
    role: str | None = Field(None, description="Snowflake role")
    session_parameters: dict[str, Any] | None = Field(
        default_factory=dict, description="Session parameters"
    )

    @property
    def has_auth(self) -> bool:
        return bool(self.password or self.private_key_path)

    def model_post_init(self, *args, **kwargs):
        if not self.has_auth:
            raise ValueError("Either password or private_key_path must be provided")


class SnowflakeSearchToolInput(BaseModel):
    """Input schema for SnowflakeSearchTool."""

    model_config = ConfigDict(protected_namespaces=())

    query: str = Field(..., description="SQL query or semantic search query to execute")
    database: str | None = Field(None, description="Override default database")
    snowflake_schema: str | None = Field(None, description="Override default schema")
    timeout: int | None = Field(300, description="Query timeout in seconds")


class SnowflakeSearchTool(BaseTool):
    """Tool for executing queries and semantic search on Snowflake."""

    name: str = "Snowflake Database Search"
    description: str = (
        "Execute SQL queries or semantic search on Snowflake data warehouse. "
        "Supports both raw SQL and natural language queries."
    )
    args_schema: type[BaseModel] = SnowflakeSearchToolInput

    # Define Pydantic fields
    config: SnowflakeConfig = Field(
        ..., description="Snowflake connection configuration"
    )
    pool_size: int = Field(default=5, description="Size of connection pool")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
    enable_caching: bool = Field(
        default=True, description="Enable query result caching"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )

    _connection_pool: list[SnowflakeConnection] | None = None
    _pool_lock: asyncio.Lock | None = None
    _thread_pool: ThreadPoolExecutor | None = None
    _model_rebuilt: bool = False
    package_dependencies: list[str] = Field(
        default_factory=lambda: [
            "snowflake-connector-python",
            "snowflake-sqlalchemy",
            "cryptography",
        ]
    )

    def __init__(self, **data):
        """Initialize SnowflakeSearchTool."""
        super().__init__(**data)
        self._initialize_snowflake()

    def _initialize_snowflake(self) -> None:
        try:
            if SNOWFLAKE_AVAILABLE:
                self._connection_pool = []
                self._pool_lock = asyncio.Lock()
                self._thread_pool = ThreadPoolExecutor(max_workers=self.pool_size)
            else:
                raise ImportError
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'snowflake-connector-python' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(
                        [  # noqa: S607
                            "uv",
                            "add",
                            "cryptography",
                            "snowflake-connector-python",
                            "snowflake-sqlalchemy",
                        ],
                        check=True,
                    )

                    self._connection_pool = []
                    self._pool_lock = asyncio.Lock()
                    self._thread_pool = ThreadPoolExecutor(max_workers=self.pool_size)
                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install Snowflake dependencies") from e
            else:
                raise ImportError(
                    "Snowflake dependencies not found. Please install them by running "
                    "`uv add cryptography snowflake-connector-python snowflake-sqlalchemy`"
                ) from None

    async def _get_connection(self) -> SnowflakeConnection:
        """Get a connection from the pool or create a new one."""
        if self._pool_lock is None:
            raise RuntimeError("Pool lock not initialized")
        if self._connection_pool is None:
            raise RuntimeError("Connection pool not initialized")
        async with self._pool_lock:
            if not self._connection_pool:
                conn = await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, self._create_connection
                )
                self._connection_pool.append(conn)
            return self._connection_pool.pop()

    def _create_connection(self) -> SnowflakeConnection:
        """Create a new Snowflake connection."""
        conn_params: dict[str, Any] = {
            "account": self.config.account,
            "user": self.config.user,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.snowflake_schema,
            "role": self.config.role,
            "session_parameters": self.config.session_parameters,
        }

        if self.config.password:
            conn_params["password"] = self.config.password.get_secret_value()
        elif self.config.private_key_path and serialization:
            with open(self.config.private_key_path, "rb") as key_file:
                p_key = serialization.load_pem_private_key(
                    key_file.read(), password=None, backend=default_backend()
                )
            conn_params["private_key"] = p_key

        return snowflake.connector.connect(**conn_params)

    def _get_cache_key(self, query: str, timeout: int) -> str:
        """Generate a cache key for the query."""
        return f"{self.config.account}:{self.config.database}:{self.config.snowflake_schema}:{query}:{timeout}"

    async def _execute_query(
        self, query: str, timeout: int = 300
    ) -> list[dict[str, Any]]:
        """Execute a query with retries and return results."""
        if self.enable_caching:
            cache_key = self._get_cache_key(query, timeout)
            if cache_key in _query_cache:
                logger.info("Returning cached result")
                return _query_cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                conn = await self._get_connection()
                try:
                    cursor = conn.cursor()
                    cursor.execute(query, timeout=timeout)

                    if not cursor.description:
                        return []

                    columns = [col[0] for col in cursor.description]
                    results = [
                        dict(zip(columns, row, strict=False))
                        for row in cursor.fetchall()
                    ]

                    if self.enable_caching:
                        _query_cache[self._get_cache_key(query, timeout)] = results

                    return results
                finally:
                    cursor.close()
                    if (
                        self._pool_lock is not None
                        and self._connection_pool is not None
                    ):
                        async with self._pool_lock:
                            self._connection_pool.append(conn)
            except (DatabaseError, OperationalError) as e:  # noqa: PERF203
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2**attempt))
                logger.warning(f"Query failed, attempt {attempt + 1}: {e!s}")
                continue
        raise RuntimeError("Query failed after all retries")

    async def _run(
        self,
        query: str,
        database: str | None = None,
        snowflake_schema: str | None = None,
        timeout: int = 300,
        **kwargs: Any,
    ) -> Any:
        """Execute the search query."""
        try:
            # Override database/schema if provided
            if database:
                await self._execute_query(f"USE DATABASE {database}")
            if snowflake_schema:
                await self._execute_query(f"USE SCHEMA {snowflake_schema}")

            return await self._execute_query(query, timeout)
        except Exception as e:
            logger.error(f"Error executing query: {e!s}")
            raise

    def __del__(self):
        """Cleanup connections on deletion."""
        try:
            if self._connection_pool:
                for conn in self._connection_pool:
                    try:
                        conn.close()
                    except Exception:  # noqa: PERF203, S110
                        pass
            if self._thread_pool:
                self._thread_pool.shutdown()
        except Exception:  # noqa: S110
            pass


try:
    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(SnowflakeSearchTool, "_model_rebuilt"):
        SnowflakeSearchTool.model_rebuild()
        SnowflakeSearchTool._model_rebuilt = True
except Exception:  # noqa: S110
    pass

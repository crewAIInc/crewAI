import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, ConfigDict, Field, SecretStr

if TYPE_CHECKING:
    # Import types for type checking only
    from snowflake.connector.connection import SnowflakeConnection
    from snowflake.connector.errors import DatabaseError, OperationalError

try:
    import snowflake.connector
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization

    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Cache for query results
_query_cache = {}


class SnowflakeConfig(BaseModel):
    """Configuration for Snowflake connection."""

    model_config = ConfigDict(protected_namespaces=())

    account: str = Field(
        ..., description="Snowflake account identifier", pattern=r"^[a-zA-Z0-9\-_]+$"
    )
    user: str = Field(..., description="Snowflake username")
    password: Optional[SecretStr] = Field(None, description="Snowflake password")
    private_key_path: Optional[str] = Field(
        None, description="Path to private key file"
    )
    warehouse: Optional[str] = Field(None, description="Snowflake warehouse")
    database: Optional[str] = Field(None, description="Default database")
    snowflake_schema: Optional[str] = Field(None, description="Default schema")
    role: Optional[str] = Field(None, description="Snowflake role")
    session_parameters: Optional[Dict[str, Any]] = Field(
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
    database: Optional[str] = Field(None, description="Override default database")
    snowflake_schema: Optional[str] = Field(None, description="Override default schema")
    timeout: Optional[int] = Field(300, description="Query timeout in seconds")


class SnowflakeSearchTool(BaseTool):
    """Tool for executing queries and semantic search on Snowflake."""

    name: str = "Snowflake Database Search"
    description: str = (
        "Execute SQL queries or semantic search on Snowflake data warehouse. "
        "Supports both raw SQL and natural language queries."
    )
    args_schema: Type[BaseModel] = SnowflakeSearchToolInput

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

    _connection_pool: Optional[List["SnowflakeConnection"]] = None
    _pool_lock: Optional[asyncio.Lock] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _model_rebuilt: bool = False

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
                        [
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
                except subprocess.CalledProcessError:
                    raise ImportError("Failed to install Snowflake dependencies")
            else:
                raise ImportError(
                    "Snowflake dependencies not found. Please install them by running "
                    "`uv add cryptography snowflake-connector-python snowflake-sqlalchemy`"
                )

    async def _get_connection(self) -> "SnowflakeConnection":
        """Get a connection from the pool or create a new one."""
        async with self._pool_lock:
            if not self._connection_pool:
                conn = await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, self._create_connection
                )
                self._connection_pool.append(conn)
            return self._connection_pool.pop()

    def _create_connection(self) -> "SnowflakeConnection":
        """Create a new Snowflake connection."""
        conn_params = {
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
    ) -> List[Dict[str, Any]]:
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
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]

                    if self.enable_caching:
                        _query_cache[self._get_cache_key(query, timeout)] = results

                    return results
                finally:
                    cursor.close()
                    async with self._pool_lock:
                        self._connection_pool.append(conn)
            except (DatabaseError, OperationalError) as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (2**attempt))
                logger.warning(f"Query failed, attempt {attempt + 1}: {str(e)}")
                continue

    async def _run(
        self,
        query: str,
        database: Optional[str] = None,
        snowflake_schema: Optional[str] = None,
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

            results = await self._execute_query(query, timeout)
            return results
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    def __del__(self):
        """Cleanup connections on deletion."""
        try:
            if self._connection_pool:
                for conn in self._connection_pool:
                    try:
                        conn.close()
                    except Exception:
                        pass
            if self._thread_pool:
                self._thread_pool.shutdown()
        except Exception:
            pass


try:
    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(SnowflakeSearchTool, "_model_rebuilt"):
        SnowflakeSearchTool.model_rebuild()
        SnowflakeSearchTool._model_rebuilt = True
except Exception:
    pass

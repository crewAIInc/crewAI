from __future__ import annotations

import importlib
import json
import os
import re
import decimal
import datetime
from collections.abc import Callable
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString


class DB2JSONEncoder(json.JSONEncoder):
    """Safely handles Decimal, Timestamps, and Bytes from DB2."""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return "<binary_data>"
        return super().default(obj)


class DB2ToolSchema(BaseModel):
    """Input schema for DB2 vector search."""

    query: str = Field(
        ...,
        description="Query to search in IBM DB2 vector database - always required.",
    )

    filter_by: str | None = Field(
        default=None,
        description=(
            "Column name used for metadata filtering. "
            "Must be used together with filter_value."
        ),
    )

    filter_value: Any | None = Field(
        default=None,
        description=(
            "Value used for metadata filtering. "
            "Must be used together with filter_by."
        ),
    )

    @model_validator(mode="after")
    def _validate_filter_pair(self) -> "DB2ToolSchema":
        if self.filter_by is not None and not self.filter_by.strip():
            raise ValueError("filter_by must be a non-empty column name.")
        if (self.filter_by is None) ^ (self.filter_value is None):
            raise ValueError("filter_by and filter_value must be provided together.")
        return self


class DB2Config(BaseModel):
    """All DB2 connection and search settings."""

    database: str
    hostname: str = "localhost"
    port: int = 50000
    protocol: str = "TCPIP"
    username: str | None = None
    password: str | None = None
    table_name: str = "documents"
    vector_column: str = "embedding"
    
    # Configure the embedding model dynamically
    embedding_model: str = "text-embedding-3-large"

    # List of columns to return from the database
    return_columns: list[str] = Field(
        default_factory=lambda: ["content"]
    )

    @model_validator(mode="after")
    def _validate_return_columns(self) -> "DB2Config":
        if not self.return_columns:
            raise ValueError(
                "return_columns cannot be empty. At least one column must be specified "
                "for the SELECT query to be valid."
            )
        return self

    # Strictly bound the limit
    limit: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of documents to return. Must be between 1 and 100."
    )
    
    # Distance metric to use (validated internally)
    distance_metric: str = "COSINE"
    
    # Prevent negative maximum distances
    max_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowed distance for results. Cannot be negative."
    )


class DB2VectorSearchTool(BaseTool):
    """
    Fortified IBM DB2 Vector Search Tool.
    Includes SQL injection protection, dynamic relational support, and type-safe serialization.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    name: str = "DB2VectorSearchTool"
    description: str = "Search IBM DB2 vector database for relevant documents."
    args_schema: type[BaseModel] = DB2ToolSchema

    # Internal Whitelist for distance metrics to prevent SQL injection
    _ALLOWED_METRICS: set[str] = {"COSINE", "EUCLIDEAN", "DOT_PRODUCT", "L2_DISTANCE"}

    package_dependencies: list[str] = Field(
        default_factory=lambda: [
            "ibm_db",
            "openai", # Optional openai is used for embeddings
        ]
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="OpenAI API key for embeddings.",
                required=False,
            ),
            EnvVar(
                name="DB2_DATABASE",
                description="IBM DB2 database name.",
                required=False,
            ),
            EnvVar(
                name="DB2_HOSTNAME",
                description="IBM DB2 hostname.",
                required=False,
            ),
            EnvVar(
                name="DB2_USERNAME",
                description="IBM DB2 username.",
                required=False,
            ),
            EnvVar(
                name="DB2_PASSWORD",
                description="IBM DB2 password.",
                required=False,
            ),
        ]
    )

    db2_config: DB2Config

    db2_package: ImportString[Any] = Field(
        default="ibm_db",
        description="IBM DB2 base package.",
    )

    db2_dbi_package: ImportString[Any] = Field(
        default="ibm_db_dbi",
        description="IBM DB2 DBI package.",
    )

    custom_embedding_fn: (
        ImportString[Callable[[str], list[float]]] | None
    ) = Field(
        default=None,
        description="Optional custom embedding function.",
    )

    connection: Any | None = None
    dbi_connection: Any | None = None
    cursor: Any | None = None

    @model_validator(mode="after")
    def _setup_db2(self) -> "DB2VectorSearchTool":
        if isinstance(self.db2_package, str):
            self.db2_package = importlib.import_module(self.db2_package)
        if isinstance(self.db2_dbi_package, str):
            self.db2_dbi_package = importlib.import_module(self.db2_dbi_package)
        return self

    def _build_connection_string(self) -> str:
        config = self.db2_config
        if config.hostname == "localhost" and not config.username:
            return config.database

        return (
            f"DATABASE={config.database};"
            f"HOSTNAME={config.hostname};"
            f"PORT={config.port};"
            f"PROTOCOL={config.protocol};"
            f"UID={config.username};"
            f"PWD={config.password};"
        )

    def _connect(self) -> None:
        if not self.connection:
            conn_str = self._build_connection_string()
            self.connection = self.db2_package.connect(conn_str, "", "")
            self.dbi_connection = self.db2_dbi_package.Connection(self.connection)
            self.cursor = self.dbi_connection.cursor()

    def _disconnect(self) -> None:
        try:
            if self.cursor:
                self.cursor.close()
            if self.dbi_connection:
                self.dbi_connection.close()
            if self.connection:
                self.db2_package.close(self.connection)
        finally:
            self.connection = None
            self.dbi_connection = None
            self.cursor = None

    def _validate_identifier(self, name: str, allow_period: bool = False) -> str:
        """
        Validates table and column names to prevent SQL injection.
        Regex allows alphanumeric, underscores, and optionally periods for schema.table.
        """
        pattern = r'^[a-zA-Z0-9_.]+$' if allow_period else r'^[a-zA-Z0-9_]+$'
        if not re.match(pattern, name):
            raise ValueError(f"Security Alert: Invalid database identifier detected: {name}")
        return name

    def _generate_embedding(self, text: str) -> list[float]:
        if self.custom_embedding_fn:
            return self.custom_embedding_fn(text)
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is missing. Required for default embeddings.")
        
        return (
            __import__("openai")
            .OpenAI(api_key=api_key)
            .embeddings.create(
                input=[text],
                model=self.db2_config.embedding_model,
            )
            .data[0]
            .embedding
        )

    def _run(
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
    ) -> str:
        # Validate query is not blank or whitespace-only
        if query is None or query.strip() == "":
            return json.dumps(
                {
                    "success": False,
                    "error": "Query cannot be empty or contain only whitespace.",
                },
                indent=2,
            )

        try:
            query_vector = self._generate_embedding(query)
            
            # Explicit Connection Handling
            try:
                self._connect()
            except Exception as e:
                self._disconnect()  # Clean up any partial connection
                return json.dumps({"success": False, "error": f"Failed to connect to DB2: {str(e)}"})

            config = self.db2_config

            # Validate Metric
            metric = config.distance_metric.upper()
            if metric not in self._ALLOWED_METRICS:
                raise ValueError(f"Invalid distance metric: {metric}")

            # Validate Identifiers
            table = self._validate_identifier(config.table_name, allow_period=True)
            v_col = self._validate_identifier(config.vector_column)
            ret_cols = [self._validate_identifier(c) for c in config.return_columns]
            
            vector_dimension = len(query_vector)
            vector_string = str(query_vector)
            
            filter_clause = ""
            params = [vector_string] # The vector string for the CLOB cast
            
            if filter_by and filter_value is not None:
                f_col = self._validate_identifier(filter_by)
                filter_clause = f"WHERE {f_col} = ?"
                params.append(filter_value)

            # DYNAMIC COLUMN SELECTION
            column_query = ", ".join(ret_cols)

            sql = f"""
                SELECT
                    {column_query},
                    VECTOR_DISTANCE(
                        {v_col},
                        VECTOR(
                            CAST(? AS CLOB),
                            {vector_dimension},
                            FLOAT32
                        ),
                        {metric}
                    ) AS distance
                FROM {table}
                {filter_clause}
                ORDER BY distance ASC
                FETCH FIRST {config.limit} ROWS ONLY
            """

            self.cursor.execute(sql, tuple(params))
            rows = self.cursor.fetchall()

            normalized_results = []

            for row in rows:
                # The 'distance' is always the LAST column in our dynamic SELECT
                distance = float(row[-1])

                if config.max_distance is not None and distance > config.max_distance:
                    continue

                # Automatically map the requested columns to their row values
                row_data = dict(zip(config.return_columns, row[:-1]))

                normalized_results.append(
                    {
                        "distance": distance,
                        "data": row_data,
                    }
                )

            # Explicit cleanup
            self._disconnect()

            return json.dumps(
                {
                    "success": True,
                    "results": normalized_results,
                },
                indent=2,
                cls=DB2JSONEncoder
            )

        except Exception as error:
            self._disconnect()
            return json.dumps(
                {
                    "success": False,
                    "error": str(error),
                    "error_type": type(error).__name__,
                },
                indent=2,
            )

    def __del__(self):
        self._disconnect()
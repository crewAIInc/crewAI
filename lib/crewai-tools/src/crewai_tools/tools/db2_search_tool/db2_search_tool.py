from __future__ import annotations

from collections.abc import Callable
import datetime
import decimal
import importlib
import json
import os
import re
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString


class DB2JSONEncoder(json.JSONEncoder):
    """Safely handles Decimal, Timestamps, and Bytes from DB2."""

    def default(self, obj: object) -> object:
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
            "Value used for metadata filtering. Must be used together with filter_by."
        ),
    )

    @model_validator(mode="after")
    def _validate_filter_pair(self) -> DB2ToolSchema:
        if self.filter_by is not None and not self.filter_by.strip():
            raise ValueError("filter_by must be a non-empty column name.")
        if (self.filter_by is None) ^ (self.filter_value is None):
            raise ValueError("filter_by and filter_value must be provided together.")
        return self


class DB2VectorSearchTool(BaseTool):
    """
    Fortified IBM DB2 Vector Search Tool.
    Includes SQL injection protection, dynamic relational support, and type-safe serialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "DB2VectorSearchTool"
    description: str = "Search IBM DB2 vector database for relevant documents. Uses a custom embedding function if supplied, otherwise OpenAI embeddings."
    args_schema: type[BaseModel] = DB2ToolSchema

    # Internal Whitelist for distance metrics to prevent SQL injection
    # Aligned with Db2 VECTOR_DISTANCE API:
    # https://www.ibm.com/docs/en/db2/12.1.x?topic=functions-vector-distance
    _ALLOWED_METRICS: ClassVar[set[str]] = {
        "COSINE",
        "EUCLIDEAN",
        "EUCLIDEAN_SQUARED",
        "DOT",
        "HAMMING",
        "MANHATTAN",
    }

    package_dependencies: list[str] = Field(
        default_factory=lambda: [
            "ibm_db",
            "openai",  # Optional openai is used for embeddings
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
                name="DB2_CONNECTION_STRING",
                description="IBM DB2 connection string (e.g. 'DATABASE=mydb;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=user;PWD=pass;').",
                required=False,
            ),
        ]
    )

    connection_string: str = Field(
        description=(
            "IBM DB2 connection string. "
            "Format: 'DATABASE=mydb;HOSTNAME=localhost;PORT=50000;PROTOCOL=TCPIP;UID=user;PWD=pass;' "
            "or just the database name for a local connection."
        )
    )

    # Search settings
    table_name: str = "documents"
    vector_column: str = "embedding"
    embedding_model: str = "text-embedding-3-large"

    return_columns: list[str] = Field(default_factory=lambda: ["content"])

    limit: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of documents to return. Must be between 1 and 100.",
    )

    distance_metric: str = "COSINE"

    max_distance: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum allowed distance for results. Cannot be negative.",
    )

    @model_validator(mode="after")
    def _validate_return_columns(self) -> DB2VectorSearchTool:
        if not self.return_columns:
            raise ValueError(
                "return_columns cannot be empty. At least one column must be specified "
                "for the SELECT query to be valid."
            )
        return self

    db2_package: Any = Field(default=None, description="IBM DB2 base package.")
    db2_dbi_package: Any = Field(default=None, description="IBM DB2 DBI package.")

    custom_embedding_fn: ImportString[Callable[[str], list[float]]] | None = Field(
        default=None,
        description="Optional custom embedding function.",
    )

    connection: Any | None = None
    dbi_connection: Any | None = None
    cursor: Any | None = None
    _openai_client: Any | None = None

    def _resolve_db2_packages(self) -> None:
        """Lazily resolve IBM DB2 packages on first use.

        Handles both default None values and explicit string inputs
        (e.g. db2_package="ibm_db") so the field always ends up as
        the real module object before _connect() uses it.
        """
        if self.db2_package is None or isinstance(self.db2_package, str):
            pkg_name = self.db2_package or "ibm_db"
            self.db2_package = importlib.import_module(pkg_name)
        if self.db2_dbi_package is None or isinstance(self.db2_dbi_package, str):
            pkg_name = self.db2_dbi_package or "ibm_db_dbi"
            self.db2_dbi_package = importlib.import_module(pkg_name)

    def _connect(self) -> None:
        self._resolve_db2_packages()
        self.connection = self.db2_package.connect(self.connection_string, "", "")
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
        Simple identifiers must start with a letter and contain only letters, digits,
        or underscores. Schema-qualified names (allow_period=True) allow exactly one
        period separating two valid simple identifiers (e.g. myschema.mytable).
        """
        pattern = (
            r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)?$"
            if allow_period
            else r"^[A-Za-z][A-Za-z0-9_]*$"
        )
        if not re.match(pattern, name):
            raise ValueError(
                f"Security Alert: Invalid database identifier detected: {name}"
            )
        return name

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is missing. Required for default embeddings."
                )
            openai = importlib.import_module("openai")
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

    def _generate_embedding(self, text: str) -> list[float]:
        if self.custom_embedding_fn:
            return self.custom_embedding_fn(text)

        result = (
            self._get_openai_client()
            .embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        return list(result)

    def _build_sql(
        self,
        column_query: str,
        v_col: str,
        vector_dimension: int,
        metric: str,
        table: str,
        filter_clause: str,
    ) -> str:
        parts = [
            "SELECT " + column_query + ",",
            " VECTOR_DISTANCE("
            + v_col
            + ", VECTOR(CAST(? AS CLOB), "
            + str(vector_dimension)
            + ", FLOAT32), "
            + metric
            + ") AS distance",
            " FROM " + table,
            " " + filter_clause if filter_clause else "",
            " ORDER BY distance ASC",
            " FETCH FIRST " + str(self.limit) + " ROWS ONLY",
        ]
        return "".join(parts)

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
                return json.dumps(
                    {"success": False, "error": f"Failed to connect to DB2: {e!s}"}
                )

            # Validate Metric
            metric = self.distance_metric.upper()
            if metric not in self._ALLOWED_METRICS:
                raise ValueError(f"Invalid distance metric: {metric}")

            # Validate Identifiers
            table = self._validate_identifier(self.table_name, allow_period=True)
            v_col = self._validate_identifier(self.vector_column)
            ret_cols = [self._validate_identifier(c) for c in self.return_columns]

            vector_dimension = len(query_vector)
            vector_string = str(query_vector)

            filter_clause = ""
            params = [vector_string]  # The vector string for the CLOB cast

            if filter_by and filter_value is not None:
                f_col = self._validate_identifier(filter_by)
                filter_clause = f"WHERE {f_col} = ?"
                params.append(filter_value)

            # DYNAMIC COLUMN SELECTION
            column_query = ", ".join(ret_cols)

            sql = self._build_sql(
                column_query, v_col, vector_dimension, metric, table, filter_clause
            )

            assert self.cursor is not None  # noqa: S101
            self.cursor.execute(sql, tuple(params))
            rows = self.cursor.fetchall()

            normalized_results = []

            for row in rows:
                # The 'distance' is always the LAST column in our dynamic SELECT
                distance = float(row[-1])

                if self.max_distance is not None and distance > self.max_distance:
                    continue

                # Automatically map the requested columns to their row values
                row_data = dict(zip(self.return_columns, row[:-1], strict=False))

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
                cls=DB2JSONEncoder,
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

    def __del__(self) -> None:
        self._disconnect()

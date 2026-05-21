from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString


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

    content_column: str = "content"

    metadata_column: str = "metadata"

    limit: int = 3

    distance_metric: str = "COSINE"

    score_threshold: float | None = 0.35


class DB2VectorSearchTool(BaseTool):
    """
    IBM DB2 Vector Search Tool.

    Supports:
    - IBM DB2 native VECTOR search
    - Optional metadata filtering
    - Custom embedding functions
    - OpenAI embedding fallback
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    name: str = "DB2VectorSearchTool"

    description: str = (
        "Search IBM DB2 vector database for relevant documents."
    )

    args_schema: type[BaseModel] = DB2ToolSchema

    package_dependencies: list[str] = Field(
        default_factory=lambda: [
            "ibm_db",
            "ibm_db_dbi",
            "openai",
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
            self.db2_package = importlib.import_module(
                self.db2_package
            )

        if isinstance(self.db2_dbi_package, str):
            self.db2_dbi_package = importlib.import_module(
                self.db2_dbi_package
            )

        return self

    def _build_connection_string(self) -> str:

        config = self.db2_config

        if (
            config.hostname == "localhost"
            and not config.username
        ):
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

            self.connection = self.db2_package.connect(
                conn_str,
                "",
                "",
            )

            self.dbi_connection = (
                self.db2_dbi_package.Connection(
                    self.connection
                )
            )

            self.cursor = self.dbi_connection.cursor()

    def _disconnect(self) -> None:

        try:

            if self.cursor:
                self.cursor.close()

            if self.dbi_connection:
                self.dbi_connection.close()

            if self.connection:
                self.db2_package.close(
                    self.connection
                )

        finally:
            self.connection = None
            self.dbi_connection = None
            self.cursor = None

    def _generate_embedding(
        self,
        text: str,
    ) -> list[float]:

        if self.custom_embedding_fn:
            return self.custom_embedding_fn(text)

        
        return (
            __import__("openai")
            .OpenAI(
                api_key=os.getenv(
                    "OPENAI_API_KEY"
                )
            )
            .embeddings.create(
                input=[text],
                model="text-embedding-3-large",
            )
            .data[0]
            .embedding
        )

    def _build_filter_clause(
        self,
        filter_by: str | None,
    ) -> str:

        if not filter_by:
            return ""

        return f"WHERE {filter_by} = ?"

    def _run(
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
    ) -> str:

        try:

            query_vector = (
                self._generate_embedding(
                    query
                )
            )

            self._connect()

            config = self.db2_config

            vector_dimension = len(query_vector)

            vector_string = str(query_vector)

            filter_clause = (
                self._build_filter_clause(
                    filter_by
                )
            )

            sql = f"""
                SELECT
                    {config.content_column},
                    {config.metadata_column},
                    VECTOR_DISTANCE(
                        {config.vector_column},
                        VECTOR(
                            '{vector_string}',
                            {vector_dimension},
                            FLOAT32
                        ),
                        {config.distance_metric}
                    ) AS distance
                FROM {config.table_name}
                {filter_clause}
                ORDER BY distance ASC
                FETCH FIRST {config.limit} ROWS ONLY
            """

            if (
                filter_by
                and filter_value is not None
            ):
                self.cursor.execute(
                    sql,
                    (filter_value,),
                )
            else:
                self.cursor.execute(sql)

            rows = self.cursor.fetchall()

            normalized_results = []

            for row in rows:

                distance = float(row[2])

                similarity = 1.0 - distance

                if (
                    config.score_threshold
                    is not None
                    and similarity
                    < config.score_threshold
                ):
                    continue

                normalized_results.append(
                    {
                        "distance": distance,
                        "metadata": (
                            json.loads(row[1])
                            if row[1]
                            else {}
                        ),
                        "context": row[0],
                    }
                )

            self._disconnect()

            return json.dumps(
                normalized_results,
                indent=2,
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
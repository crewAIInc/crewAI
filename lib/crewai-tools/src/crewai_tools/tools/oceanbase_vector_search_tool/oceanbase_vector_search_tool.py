from __future__ import annotations

from collections.abc import Callable
import importlib
import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString



class OceanBaseToolSchema(BaseModel):
    query: str = Field(
        ..., description="Query to search in OceanBase vector database - always required."
    )
    filter_by: str | None = Field(
        default=None,
        description="Column name to filter the search by. When filtering, needs to be used in conjunction with filter_value.",
    )
    filter_value: Any | None = Field(
        default=None,
        description="Value to filter the search by. When filtering, needs to be used in conjunction with filter_by.",
    )


class OceanBaseConfig(BaseModel):
    """All OceanBase connection and search settings."""

    uri: str = Field(..., description="OceanBase connection URI (e.g., '127.0.0.1:2881')")
    user: str = Field(..., description="OceanBase user (e.g., 'root@test')")
    password: str = Field(default="", description="OceanBase password")
    db_name: str = Field(..., description="OceanBase database name")
    table_name: str = Field(..., description="Table name to search in")
    vec_column_name: str = Field(
        default="embedding", description="Vector column name in the table"
    )
    limit: int = Field(default=3, description="Number of results to return")
    distance_threshold: float | None = Field(
        default=None, description="Distance threshold to filter results"
    )
    distance_func: str = Field(
        default="l2_distance",
        description="Distance function: 'l2_distance', 'cosine_distance', 'inner_product', or 'negative_inner_product'",
    )
    output_columns: list[str] | None = Field(
        default=None, description="List of column names to return in results"
    )


class OceanBaseVectorSearchTool(BaseTool):
    """Vector search tool for OceanBase."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Metadata ---
    name: str = "OceanBaseVectorSearchTool"
    description: str = "Search OceanBase vector database for relevant documents using vector similarity search."
    args_schema: type[BaseModel] = OceanBaseToolSchema
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["pyobvector", "numpy"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI (used for embeddings if custom_embedding_fn not provided)",
                required=False,
            )
        ]
    )
    oceanbase_config: OceanBaseConfig
    pyobvector_package: ImportString[Any] = Field(
        default="pyobvector",
        description="Base package path for pyobvector. Will dynamically import client and distance functions.",
    )
    custom_embedding_fn: ImportString[Callable[[str], list[float]]] | None = Field(
        default=None,
        description="Optional embedding function or import path.",
    )
    client: Any | None = None

    @model_validator(mode="after")
    def _setup_oceanbase(self) -> OceanBaseVectorSearchTool:
        # Import the pyobvector package if it's a string
        if isinstance(self.pyobvector_package, str):
            self.pyobvector_package = importlib.import_module(self.pyobvector_package)

        if not self.client:
            # Import ObVecClient and distance functions
            ObVecClient = getattr(self.pyobvector_package, "ObVecClient")
            self.client = ObVecClient(
                uri=self.oceanbase_config.uri,
                user=self.oceanbase_config.user,
                password=self.oceanbase_config.password,
                db_name=self.oceanbase_config.db_name,
            )
        return self

    def __del__(self) -> None:
        """Close database connection on deletion to avoid connection pool exhaustion."""
        try:
            if hasattr(self, "client") and self.client is not None:
                if callable(getattr(self.client, "close", None)):
                    self.client.close()
                elif hasattr(self.client, "engine") and hasattr(
                    self.client.engine, "dispose"
                ):
                    self.client.engine.dispose()
        except Exception:  # noqa: S110
            pass

    def _get_distance_func(self):
        """Get the distance function based on configuration."""
        distance_func_name = self.oceanbase_config.distance_func
        distance_func_map = {
            "l2_distance": getattr(self.pyobvector_package, "l2_distance"),
            "cosine_distance": getattr(self.pyobvector_package, "cosine_distance"),
            "inner_product": getattr(self.pyobvector_package, "inner_product"),
            "negative_inner_product": getattr(
                self.pyobvector_package, "negative_inner_product"
            ),
        }

        if distance_func_name not in distance_func_map:
            raise ValueError(
                f"Unknown distance function: {distance_func_name}. "
                f"Supported functions: {list(distance_func_map.keys())}"
            )

        return distance_func_map[distance_func_name]

    def _build_where_clause(self, filter_by: str | None, filter_value: Any | None):
        """Build SQLAlchemy where clause for filtering."""
        if not filter_by or filter_value is None:
            return None

        # Import Table here to avoid import errors if sqlalchemy is not available
        # (though it should be available via pyobvector dependency)
        try:
            from sqlalchemy import Table
        except ImportError:
            raise ImportError(
                "sqlalchemy is required for OceanBaseVectorSearchTool. "
                "It should be installed as a dependency of pyobvector."
            )

        table = Table(
            self.oceanbase_config.table_name,
            self.client.metadata_obj,
            autoload_with=self.client.engine,
        )

        if filter_by not in table.columns:
            raise ValueError(
                f"Column '{filter_by}' not found in table '{self.oceanbase_config.table_name}'"
            )

        return [table.c[filter_by] == filter_value]

    def _run(
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
    ) -> str:
        """Perform vector similarity search."""

        # Get embedding vector for the query
        query_vector = (
            self.custom_embedding_fn(query)
            if self.custom_embedding_fn
            else (
                lambda: __import__("openai")
                .Client(api_key=os.getenv("OPENAI_API_KEY"))
                .embeddings.create(input=[query], model="text-embedding-3-large")
                .data[0]
                .embedding
            )()
        )

        # Get distance function
        distance_func = self._get_distance_func()

        # Build where clause if filtering is requested
        where_clause = self._build_where_clause(filter_by, filter_value)

        # Perform ANN search
        results = self.client.ann_search(
            table_name=self.oceanbase_config.table_name,
            vec_data=query_vector,
            vec_column_name=self.oceanbase_config.vec_column_name,
            distance_func=distance_func,
            with_dist=True,
            topk=self.oceanbase_config.limit,
            output_column_names=self.oceanbase_config.output_columns,
            where_clause=where_clause,
            distance_threshold=self.oceanbase_config.distance_threshold,
        )

        # Format results
        formatted_results = []

        # Get table schema for column names
        # Import Table here to avoid import errors
        try:
            from sqlalchemy import Table
        except ImportError:
            raise ImportError(
                "sqlalchemy is required for OceanBaseVectorSearchTool. "
                "It should be installed as a dependency of pyobvector."
            )

        table = Table(
            self.oceanbase_config.table_name,
            self.client.metadata_obj,
            autoload_with=self.client.engine,
        )

        for row in results:
            # Convert row to dict
            row_dict = {}

            # Distance is always the last column when with_dist=True
            # So we need to exclude it from column mapping
            row_data = row[:-1] if len(row) > 0 else []
            distance = row[-1] if len(row) > 0 else None

            if self.oceanbase_config.output_columns:
                # Map output columns to values
                for i, col_name in enumerate(self.oceanbase_config.output_columns):
                    if i < len(row_data):
                        row_dict[col_name] = row_data[i]
            else:
                # Get all columns from table (excluding distance)
                all_columns = [col.name for col in table.columns]
                for i, col_name in enumerate(all_columns):
                    if i < len(row_data):
                        row_dict[col_name] = row_data[i]

            # Extract text content from common column names
            text_content = ""
            for text_col in ["text", "content", "body", "description", "summary"]:
                if text_col in row_dict:
                    text_content = str(row_dict[text_col])
                    break

            # If no text column found, use first non-vector column
            if not text_content:
                for key, value in row_dict.items():
                    if key != self.oceanbase_config.vec_column_name:
                        text_content = str(value)
                        break

            formatted_results.append(
                {
                    "distance": float(distance) if distance is not None else None,
                    "metadata": {
                        k: v
                        for k, v in row_dict.items()
                        if k != self.oceanbase_config.vec_column_name
                    },
                    "context": text_content,
                }
            )

        return json.dumps(formatted_results, indent=2, default=str)

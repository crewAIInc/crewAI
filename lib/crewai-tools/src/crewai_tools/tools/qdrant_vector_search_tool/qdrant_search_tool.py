from __future__ import annotations

from collections.abc import Callable
import importlib
import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString


class QdrantToolSchema(BaseModel):
    query: str = Field(
        ..., description="Query to search in Qdrant DB - always required."
    )
    filter_by: str | None = Field(
        default=None,
        description="Parameter to filter the search by. When filtering, needs to be used in conjunction with filter_value.",
    )
    filter_value: Any | None = Field(
        default=None,
        description="Value to filter the search by. When filtering, needs to be used in conjunction with filter_by.",
    )


class QdrantConfig(BaseModel):
    """All Qdrant connection and search settings."""

    qdrant_url: str
    qdrant_api_key: str | None = None
    collection_name: str
    limit: int = 3
    score_threshold: float = 0.35
    filter: Any | None = Field(
        default=None, description="Qdrant Filter instance for advanced filtering."
    )


class QdrantVectorSearchTool(BaseTool):
    """Vector search tool for Qdrant."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Metadata ---
    name: str = "QdrantVectorSearchTool"
    description: str = "Search Qdrant vector DB for relevant documents."
    args_schema: type[BaseModel] = QdrantToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["qdrant-client"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY", description="API key for OpenAI", required=True
            )
        ]
    )
    qdrant_config: QdrantConfig
    qdrant_package: ImportString[Any] = Field(
        default="qdrant_client",
        description="Base package path for Qdrant. Will dynamically import client and models.",
    )
    custom_embedding_fn: ImportString[Callable[[str], list[float]]] | None = Field(
        default=None,
        description="Optional embedding function or import path.",
    )
    client: Any | None = None

    @model_validator(mode="after")
    def _setup_qdrant(self) -> QdrantVectorSearchTool:
        # Import the qdrant_package if it's a string
        if isinstance(self.qdrant_package, str):
            self.qdrant_package = importlib.import_module(self.qdrant_package)

        if not self.client:
            self.client = self.qdrant_package.QdrantClient(
                url=self.qdrant_config.qdrant_url,
                api_key=self.qdrant_config.qdrant_api_key or None,
            )
        return self

    def _run(
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
    ) -> str:
        """Perform vector similarity search."""

        search_filter = (
            self.qdrant_config.filter.model_copy()
            if self.qdrant_config.filter is not None
            else self.qdrant_package.http.models.Filter(must=[])
        )
        if filter_by and filter_value is not None:
            if not hasattr(search_filter, "must") or not isinstance(
                search_filter.must, list
            ):
                search_filter.must = []
            search_filter.must.append(
                self.qdrant_package.http.models.FieldCondition(
                    key=filter_by,
                    match=self.qdrant_package.http.models.MatchValue(
                        value=filter_value
                    ),
                )
            )

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
        results = self.client.query_points(
            collection_name=self.qdrant_config.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=self.qdrant_config.limit,
            score_threshold=self.qdrant_config.score_threshold,
        )

        return json.dumps(
            [
                {
                    "distance": p.score,
                    "metadata": p.payload.get("metadata", {}) if p.payload else {},
                    "context": p.payload.get("text", "") if p.payload else {},
                }
                for p in results.points
            ],
            indent=2,
        )

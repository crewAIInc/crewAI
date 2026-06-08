from __future__ import annotations

from collections.abc import Callable
import importlib
import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import ImportString


class SochDBToolSchema(BaseModel):
    query: str = Field(
        ...,
        description="Query to search in SochDB - always required.",
    )
    metadata_filter_json: str | None = Field(
        default=None,
        description="Optional JSON object used to filter documents by metadata.",
    )


class SochDBConfig(BaseModel):
    """All SochDB connection and search settings."""

    grpc_address: str
    collection_name: str
    namespace: str = "default"
    limit: int = 3


class SochDBVectorSearchTool(BaseTool):
    """Vector search tool for SochDB."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "SochDBVectorSearchTool"
    description: str = "Search SochDB for relevant documents using vector similarity."
    args_schema: type[BaseModel] = SochDBToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["sochdb"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI embeddings",
                required=True,
            )
        ]
    )
    sochdb_config: SochDBConfig
    sochdb_package: ImportString[Any] = Field(
        default="sochdb",
        description="Base package path for SochDB.",
    )
    custom_embedding_fn: ImportString[Callable[[str], list[float]]] | None = Field(
        default=None,
        description="Optional embedding function or import path.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model used when no custom embedding function is provided.",
    )
    client: Any | None = None

    @model_validator(mode="after")
    def _setup_sochdb(self) -> SochDBVectorSearchTool:
        if self.client is not None:
            return self

        if isinstance(self.sochdb_package, str):
            self.sochdb_package = importlib.import_module(self.sochdb_package)

        self.client = self.sochdb_package.SochDBClient(self.sochdb_config.grpc_address)
        return self

    def _embed_query(self, query: str) -> list[float]:
        if self.custom_embedding_fn:
            return list(self.custom_embedding_fn(query))

        return (
            __import__("openai")
            .Client(api_key=os.getenv("OPENAI_API_KEY"))
            .embeddings.create(input=[query], model=self.embedding_model)
            .data[0]
            .embedding
        )

    def _run(self, query: str, metadata_filter_json: str | None = None) -> str:
        metadata_filter = None
        if metadata_filter_json:
            parsed = json.loads(metadata_filter_json)
            if not isinstance(parsed, dict):
                raise ValueError("metadata_filter_json must decode to a JSON object")
            metadata_filter = {str(k): str(v) for k, v in parsed.items()}

        results = self.client.search_collection(  # type: ignore[union-attr]
            self.sochdb_config.collection_name,
            self._embed_query(query),
            k=self.sochdb_config.limit,
            namespace=self.sochdb_config.namespace,
            filter=metadata_filter,
        )

        return json.dumps(
            [
                {
                    "id": doc.id,
                    "metadata": doc.metadata,
                    "context": doc.content,
                }
                for doc in results
            ],
            indent=2,
        )

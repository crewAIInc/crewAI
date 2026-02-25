from __future__ import annotations

import json
from logging import getLogger
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


try:
    import pyobvector  # noqa: F401

    PYOBVECTOR_AVAILABLE = True
except ImportError:
    PYOBVECTOR_AVAILABLE = False

logger = getLogger(__name__)


class OceanBaseToolSchema(BaseModel):
    """Input schema for OceanBase vector search tool."""

    query: str = Field(
        ...,
        description="The query to search for relevant information in the OceanBase database.",
    )


class OceanBaseVectorSearchConfig(BaseModel):
    """Configuration for OceanBase vector search queries."""

    limit: int = Field(
        default=4,
        description="Number of documents to return.",
    )
    distance_threshold: float | None = Field(
        default=None,
        description="Only return results where distance is less than or equal to this threshold.",
    )
    distance_func: str = Field(
        default="l2",
        description="Distance function to use for similarity search. Options: 'l2', 'cosine', 'inner_product'.",
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include the embedding vector of each result.",
    )


class OceanBaseVectorSearchTool(BaseTool):
    """Tool to perform vector search on OceanBase database."""

    name: str = "OceanBaseVectorSearchTool"
    description: str = (
        "A tool to perform vector similarity search on an OceanBase database "
        "for retrieving relevant information from stored documents."
    )

    args_schema: type[BaseModel] = OceanBaseToolSchema
    query_config: OceanBaseVectorSearchConfig | None = Field(
        default=None,
        description="OceanBase vector search query configuration.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model to use for generating query embeddings.",
    )
    dimensions: int = Field(
        default=1536,
        description="Number of dimensions in the embedding vector.",
    )
    connection_uri: str = Field(
        ...,
        description="Connection URI for OceanBase (e.g., '127.0.0.1:2881').",
    )
    user: str = Field(
        ...,
        description="Username for OceanBase connection (e.g., 'root@test').",
    )
    password: str = Field(
        default="",
        description="Password for OceanBase connection.",
    )
    db_name: str = Field(
        default="test",
        description="Database name in OceanBase.",
    )
    table_name: str = Field(
        ...,
        description="Name of the table containing vector data.",
    )
    vector_column_name: str = Field(
        default="embedding",
        description="Name of the column containing vector embeddings.",
    )
    text_column_name: str = Field(
        default="text",
        description="Name of the column containing text content.",
    )
    metadata_column_name: str | None = Field(
        default="metadata",
        description="Name of the column containing metadata (optional).",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for OpenAI embeddings",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["pyobvector"])

    _client: Any = None
    _openai_client: Any = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not PYOBVECTOR_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'pyobvector' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "pyobvector"], check=True)  # noqa: S607
            else:
                raise ImportError(
                    "The 'pyobvector' package is required for OceanBaseVectorSearchTool."
                )

        if "AZURE_OPENAI_ENDPOINT" in os.environ:
            from openai import AzureOpenAI

            self._openai_client = AzureOpenAI()
        elif "OPENAI_API_KEY" in os.environ:
            from openai import Client

            self._openai_client = Client()
        else:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OceanBaseVectorSearchTool."
            )

        from pyobvector import ObVecClient

        self._client = ObVecClient(
            uri=self.connection_uri,
            user=self.user,
            password=self.password,
            db_name=self.db_name,
        )

    def _embed_text(self, text: str) -> list[float]:
        """Generate embedding for the given text using OpenAI."""
        response = self._openai_client.embeddings.create(
            input=[text],
            model=self.embedding_model,
            dimensions=self.dimensions,
        )
        return response.data[0].embedding

    def _get_distance_func(self) -> Any:
        """Get the appropriate distance function from pyobvector."""
        import pyobvector

        config = self.query_config or OceanBaseVectorSearchConfig()
        valid_distance_funcs = {
            "l2": "l2_distance",
            "cosine": "cosine_distance",
            "inner_product": "inner_product",
        }

        func_name = valid_distance_funcs.get(config.distance_func, "l2_distance")
        return getattr(pyobvector, func_name)

    def _run(self, query: str) -> str:
        """Execute vector search on OceanBase."""
        try:
            config = self.query_config or OceanBaseVectorSearchConfig()

            query_vector = self._embed_text(query)

            output_columns = [self.text_column_name]
            if self.metadata_column_name:
                output_columns.append(self.metadata_column_name)

            results = self._client.ann_search(
                table_name=self.table_name,
                vec_data=query_vector,
                vec_column_name=self.vector_column_name,
                distance_func=self._get_distance_func(),
                with_dist=True,
                topk=config.limit,
                output_column_names=output_columns,
                distance_threshold=config.distance_threshold,
            )

            formatted_results = []
            for row in results:
                result_dict: dict[str, Any] = {}

                if len(row) >= 1:
                    result_dict["text"] = row[0]
                if self.metadata_column_name and len(row) >= 2:
                    result_dict["metadata"] = row[1]
                if len(row) > len(output_columns):
                    result_dict["distance"] = row[-1]

                formatted_results.append(result_dict)

            return json.dumps(formatted_results, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error during OceanBase vector search: {e}")
            return json.dumps({"error": str(e)})

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add texts with embeddings to the OceanBase table.

        Args:
            texts: List of text strings to add.
            metadatas: Optional list of metadata dictionaries for each text.
            ids: Optional list of unique IDs for each text.

        Returns:
            List of IDs for the added texts.
        """
        import uuid

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{} for _ in texts]

        data = []
        for text, metadata, doc_id in zip(texts, metadatas, ids, strict=False):
            embedding = self._embed_text(text)
            row = {
                "id": doc_id,
                self.text_column_name: text,
                self.vector_column_name: embedding,
            }
            if self.metadata_column_name:
                row[self.metadata_column_name] = metadata
            data.append(row)

        self._client.insert(self.table_name, data=data)
        return ids

    def __del__(self) -> None:
        """Cleanup clients on deletion."""
        try:
            if hasattr(self, "_openai_client") and self._openai_client:
                self._openai_client.close()
        except Exception as e:
            logger.error(f"Error closing OpenAI client: {e}")

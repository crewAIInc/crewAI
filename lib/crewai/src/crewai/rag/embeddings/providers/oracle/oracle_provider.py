"""Oracle embeddings provider."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from crewai.rag.core.base_embeddings_provider import BaseEmbeddingsProvider
from crewai.rag.embeddings.providers.oracle.embedding_callable import (
    OracleEmbeddingFunction,
)


class OracleProvider(BaseEmbeddingsProvider[OracleEmbeddingFunction]):
    """Oracle embeddings provider."""

    embedding_callable: type[OracleEmbeddingFunction] = Field(
        default=OracleEmbeddingFunction,
        description="Oracle embedding function class",
    )
    conn: Any | None = Field(
        default=None,
        description="Existing Oracle connection object to reuse.",
    )
    connection_params: dict[str, Any] | None = Field(
        default=None,
        description="Keyword arguments forwarded to oracledb.connect(**connection_params).",
    )
    embedding_params: dict[str, Any] = Field(
        ...,
        description="Parameters forwarded to dbms_vector_chain.utl_to_embeddings.",
    )
    proxy: str | None = Field(
        default=None,
        description="Optional proxy value passed to utl_http.set_proxy before embedding requests.",
    )

    @model_validator(mode="after")
    def validate_connection_source(self) -> "OracleProvider":
        has_conn = self.conn is not None
        has_connection_params = self.connection_params is not None
        if has_conn == has_connection_params:
            raise ValueError(
                "Provide exactly one of 'conn' or 'connection_params' for oracle embeddings."
            )
        return self

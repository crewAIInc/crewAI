from __future__ import annotations

from collections.abc import Iterable
import math
import os
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from crewai.tools import BaseTool, EnvVar
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field, PrivateAttr


if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
    from azure.identity import DefaultAzureCredential
    from pymongo import MongoClient
    from pymongo.collection import Collection


class AzureDocumentDBVectorSearchConfig(BaseModel):
    """Options for Azure DocumentDB's native vector search."""

    limit: int = Field(default=4, ge=1, description="Maximum number of results.")
    pre_filter: dict[str, Any] | None = Field(
        default=None,
        description="DocumentDB filter applied during search, e.g. metadata.category.",
    )
    include_score: bool = Field(default=True, description="Return similarity scores.")
    include_embeddings: bool = Field(
        default=False, description="Return stored embedding vectors."
    )


class AzureDocumentDBToolSchema(BaseModel):
    """Input accepted by the Azure DocumentDB vector search tool."""

    query: str = Field(
        ..., min_length=1, description="Text to find relevant documents for."
    )


class AzureDocumentDBVectorSearchTool(BaseTool):
    """Retrieve documents using Azure DocumentDB, not MongoDB Atlas vector search.

    Documents contain ``text``, ``metadata`` and ``embedding`` fields. Embeddings
    use OpenAI, or Azure OpenAI when AZURE_OPENAI_ENDPOINT is configured.
    """

    name: str = "AzureDocumentDBVectorSearchTool"
    description: str = (
        "Search Azure DocumentDB for documents relevant to a natural language query."
    )
    args_schema: type[BaseModel] = AzureDocumentDBToolSchema
    connection_string: str = Field(
        default_factory=lambda: os.environ.get(
            "AZURE_DOCUMENTDB_CONNECTION_STRING", ""
        ),
        min_length=1,
        repr=False,
        exclude=True,
        description="Azure DocumentDB MongoDB-compatible connection string.",
    )
    database_name: str = Field(..., min_length=1)
    collection_name: str = Field(..., min_length=1)
    query_config: AzureDocumentDBVectorSearchConfig = Field(
        default_factory=AzureDocumentDBVectorSearchConfig
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="OpenAI embedding model or Azure OpenAI deployment name.",
    )
    dimensions: int = Field(default=1536, ge=1)
    vector_index_name: str = Field(default="crewai_vector_index", min_length=1)
    connection_timeout_ms: int = Field(default=10000, ge=1)
    use_azure_identity: bool = Field(
        default=False,
        description="Use renewable Entra tokens for Azure OpenAI instead of an API key.",
    )
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["pymongo", "azure-identity"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="AZURE_DOCUMENTDB_CONNECTION_STRING",
                description="Connection string, unless passed to the constructor.",
                required=False,
            ),
            EnvVar(
                name="OPENAI_API_KEY",
                description="OpenAI key (not required when using Azure OpenAI).",
                required=False,
            ),
            EnvVar(
                name="AZURE_OPENAI_ENDPOINT",
                description="Set to use Azure OpenAI for embeddings.",
                required=False,
            ),
            EnvVar(
                name="AZURE_OPENAI_API_KEY",
                description="Azure OpenAI key when using Azure OpenAI.",
                required=False,
            ),
            EnvVar(
                name="OPENAI_API_VERSION",
                description="Azure OpenAI API version when using Azure OpenAI.",
                required=False,
            ),
        ]
    )

    _client: MongoClient[dict[str, Any]] = PrivateAttr()
    _coll: Collection[dict[str, Any]] = PrivateAttr()
    _openai_client: OpenAI | AzureOpenAI | None = PrivateAttr(default=None)
    _azure_credential: TokenCredential | None = PrivateAttr(default=None)
    _owned_credential: DefaultAzureCredential | None = PrivateAttr(default=None)

    def __init__(
        self, *, azure_credential: TokenCredential | None = None, **kwargs: Any
    ) -> None:
        """Accept an optional Azure credential, whose lifetime the caller owns."""
        super().__init__(**kwargs)
        try:
            from pymongo import MongoClient
        except ImportError as exc:
            raise ImportError(
                "Install Azure DocumentDB support with "
                "`pip install 'crewai-tools[azure-documentdb]'`."
            ) from exc

        self._azure_credential = azure_credential
        auth_options: dict[str, Any] = {}
        uri_options = {
            key.lower(): value
            for key, value in parse_qsl(urlsplit(self.connection_string).query)
        }
        if uri_options.get("authmechanism", "").upper() == "MONGODB-OIDC":
            from crewai_tools.tools.azure_documentdb_vector_search_tool.auth import (
                DocumentDBOIDCCallback,
            )

            auth_options = {
                "authMechanismProperties": {
                    "OIDC_CALLBACK": DocumentDBOIDCCallback(
                        self._get_azure_credential()
                    )
                }
            }
        self._client = MongoClient(
            self.connection_string,
            appname="crewai-azure-documentdb",
            serverSelectionTimeoutMS=self.connection_timeout_ms,
            **auth_options,
        )
        self._coll = self._client[self.database_name][self.collection_name]

    def _get_azure_credential(self) -> TokenCredential:
        """Reuse the injected credential or create an internally owned default."""
        if self._azure_credential is None:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as exc:
                raise ImportError(
                    "Entra authentication requires azure-identity. Install "
                    "`pip install 'crewai-tools[azure-documentdb]'`."
                ) from exc
            self._owned_credential = DefaultAzureCredential()
            self._azure_credential = self._owned_credential
        return self._azure_credential

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate ordered, finite embeddings matching the configured dimensions."""
        if self._openai_client is None:
            if os.environ.get("AZURE_OPENAI_ENDPOINT"):
                if self.use_azure_identity or self._azure_credential is not None:
                    from azure.identity import get_bearer_token_provider

                    self._openai_client = AzureOpenAI(
                        azure_ad_token_provider=get_bearer_token_provider(
                            self._get_azure_credential(),
                            "https://cognitiveservices.azure.com/.default",
                        )
                    )
                else:
                    self._openai_client = AzureOpenAI()
            elif self.use_azure_identity:
                raise ValueError(
                    "AZURE_OPENAI_ENDPOINT is required when use_azure_identity=True."
                )
            elif os.environ.get("OPENAI_API_KEY"):
                self._openai_client = OpenAI()
            else:
                raise ValueError(
                    "Set OPENAI_API_KEY, or AZURE_OPENAI_ENDPOINT, "
                    "AZURE_OPENAI_API_KEY and OPENAI_API_VERSION for embeddings."
                )
        response = self._openai_client.embeddings.create(
            input=texts, model=self.embedding_model, dimensions=self.dimensions
        )
        embeddings = [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]
        if len(embeddings) != len(texts) or any(
            len(vector) != self.dimensions
            or not all(math.isfinite(value) for value in vector)
            for vector in embeddings
        ):
            raise ValueError(
                "Embedding response must contain one finite vector per text "
                f"with {self.dimensions} dimensions."
            )
        return embeddings

    def create_vector_search_index(
        self, *, similarity: Literal["COS", "L2", "IP"] = "COS", num_lists: int = 1
    ) -> None:
        """Create an IVF index without dropping existing indexes.

        The collection must already exist. Use the same dimensions and similarity
        for ingestion and search. Tune num_lists for larger collections.
        """
        if similarity not in {"COS", "L2", "IP"}:
            raise ValueError("similarity must be COS, L2 or IP.")
        if num_lists < 1:
            raise ValueError("num_lists must be positive.")
        self._coll.database.command(
            {
                "createIndexes": self.collection_name,
                "indexes": [
                    {
                        "name": self.vector_index_name,
                        "key": {"embedding": "cosmosSearch"},
                        "cosmosSearchOptions": {
                            "kind": "vector-ivf",
                            "numLists": num_lists,
                            "similarity": similarity,
                            "dimensions": self.dimensions,
                        },
                    }
                ],
            }
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        batch_size: int = 100,
    ) -> list[str]:
        """Embed and insert texts, returning their IDs.

        Supplied IDs are preserved; duplicates raise the database's write error.
        Metadata is nested under ``metadata`` and cannot overwrite text or vectors.
        Batches already inserted remain persisted if a later batch fails.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        documents = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in documents):
            raise ValueError("texts must contain non-blank strings.")
        if metadatas is not None and len(metadatas) != len(documents):
            raise ValueError("metadatas must have the same length as texts.")
        if ids is not None and (
            len(ids) != len(documents)
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("ids must contain one unique, non-empty string per text.")
        result_ids = ids if ids is not None else [str(uuid4()) for _ in documents]
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            embeddings = self._embed_texts(batch)
            self._coll.insert_many(
                [
                    {
                        "_id": result_ids[start + offset],
                        "text": text,
                        "metadata": metadatas[start + offset]
                        if metadatas is not None
                        else {},
                        "embedding": vector,
                    }
                    for offset, (text, vector) in enumerate(
                        zip(batch, embeddings, strict=True)
                    )
                ]
            )
        return list(result_ids)

    def _run(self, query: str) -> str:
        """Search DocumentDB and serialize matching documents as Extended JSON."""
        from bson import json_util

        if not query.strip():
            raise ValueError("query must not be blank.")
        config = self.query_config
        search: dict[str, Any] = {
            "vector": self._embed_texts([query])[0],
            "path": "embedding",
            "k": config.limit,
        }
        if config.pre_filter is not None:
            search["filter"] = config.pre_filter
        pipeline: list[dict[str, Any]] = [
            {"$search": {"cosmosSearch": search, "returnStoredSource": True}}
        ]
        if config.include_score:
            pipeline.append({"$addFields": {"score": {"$meta": "searchScore"}}})
        if not config.include_embeddings:
            pipeline.append({"$project": {"embedding": 0}})
        return str(json_util.dumps(list(self._coll.aggregate(pipeline))))

    def close(self) -> None:
        """Release clients and internally created credentials, not caller credentials."""
        self._client.close()
        if self._openai_client is not None:
            self._openai_client.close()
        if self._owned_credential is not None:
            self._owned_credential.close()

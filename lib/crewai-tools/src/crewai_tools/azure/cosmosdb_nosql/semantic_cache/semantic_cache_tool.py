"""Azure CosmosDB NoSQL semantic cache tool for LLM responses."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from logging import getLogger
from typing import Any, ClassVar

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from crewai_tools.azure.cosmosdb_nosql._client import (
    build_cosmos_client,
    close_cosmos_client,
    create_container_if_not_exists,
    create_database_if_not_exists,
)
from crewai_tools.azure.cosmosdb_nosql._embeddings import (
    build_openai_client,
    embed_texts,
    embed_texts_via_embedder,
)
from crewai_tools.azure.cosmosdb_nosql._utils import (
    DistanceStrategy,
    quote_sql_string,
    score_threshold_passes,
)


logger = getLogger(__name__)


def _quote_terms(text: str) -> str:
    """Return a SQL-safe comma-separated list of single-quoted FTS terms."""
    return ", ".join(quote_sql_string(t) for t in text.split() if t)


def _llm_namespace(llm_string: str | None) -> str:
    """Return a stable, short hash of an LLM identifier for cache namespacing."""
    if not llm_string:
        return "default"
    return hashlib.sha256(llm_string.encode("utf-8")).hexdigest()[:32]


class AzureCosmosDBSemanticCacheConfig(BaseModel):
    """Configuration for :class:`AzureCosmosDBSemanticCacheTool`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cosmos_host: str | None = Field(
        default=None, description="The CosmosDB account endpoint URL."
    )
    connection_string: str | None = Field(
        default=None,
        description="CosmosDB account connection string (alternative to host+key).",
    )
    key: str | None = Field(default=None, description="The Azure account key.")
    token_credential: Any | None = Field(
        default=None,
        description="An Azure azure.core.credentials.TokenCredential instance.",
    )
    embedder: Any | None = Field(
        default=None,
        description=(
            "Optional custom embedder (``embed_documents`` / ``embed_query``). "
            "Overrides OpenAI/Azure OpenAI when provided."
        ),
    )
    database_name: str = Field(
        default="memory_database", description="CosmosDB database name."
    )
    container_name: str = Field(
        default="memory_container", description="CosmosDB container name."
    )
    cosmos_container_properties: dict[str, Any] = Field(
        default_factory=lambda: {
            "partition_key": {"paths": ["/agent_id"], "kind": "Hash"}
        },
        description="Container properties (partition key, etc.).",
    )
    cosmos_database_properties: dict[str, Any] = Field(
        default_factory=dict, description="Database properties."
    )
    create_container: bool = Field(
        default=True,
        description="Create the container at init if it does not exist.",
    )
    vector_embedding_policy: dict[str, Any] | None = Field(
        default=None, description="Vector embedding policy for container creation."
    )
    indexing_policy: dict[str, Any] = Field(
        ..., description="Indexing policy for container creation."
    )
    full_text_policy: dict[str, Any] | None = Field(
        default=None, description="Full-text policy for container creation."
    )
    dimensions: int = Field(
        default=1536, description="Embedding vector dimensionality."
    )
    table_alias: str = Field(
        default="c", description="SQL alias for the container in queries."
    )

    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI / Azure OpenAI embedding model identifier.",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality requested from the embedding model.",
    )
    azure_openai_endpoint: str | None = Field(
        default=None, description="Azure OpenAI endpoint URL."
    )
    openai_api_key: str | None = Field(
        default=None, description="OpenAI / Azure OpenAI API key."
    )

    similarity_threshold: float = Field(
        default=0.85,
        description=(
            "Minimum similarity for a cache hit (cosine/dot-product) or maximum "
            "distance (euclidean)."
        ),
    )
    default_ttl: int | None = Field(
        default=86400,
        description="Default TTL in seconds (set to None to disable expiry).",
    )
    enable_hybrid_search: bool = Field(
        default=True,
        description="Combine vector + full-text RRF search when enabled.",
    )
    llm_string: str | None = Field(
        default=None,
        description=(
            "Identifier for the LLM whose responses are being cached "
            "(e.g. 'gpt-4o-2024-08-06'). Used to namespace cache entries so "
            "responses from different models do not collide."
        ),
    )


class AzureCosmosDBSemanticCacheToolSchema(BaseModel):
    """Input schema for :class:`AzureCosmosDBSemanticCacheTool`."""

    operation: str = Field(
        ..., description="Operation: 'search', 'update', or 'clear'."
    )
    prompt: str | None = Field(
        default=None,
        description="Prompt to look up (search) or store (update).",
    )
    response: str | None = Field(
        default=None,
        description="LLM response to associate with the prompt (update only).",
    )


class AzureCosmosDBSemanticCacheTool(BaseTool):
    """Semantic cache for LLM responses backed by Azure CosmosDB NoSQL."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    USER_AGENT: ClassVar[str] = "CrewAI-CosmosDB-SemanticCache-Tool-Python"

    name: str = "AzureCosmosDBSemanticCacheTool"
    description: str = (
        "Semantic cache for LLM responses backed by Azure CosmosDB NoSQL vector search."
    )
    args_schema: type[BaseModel] = AzureCosmosDBSemanticCacheToolSchema

    config: AzureCosmosDBSemanticCacheConfig = Field(
        ..., description="Configuration for the semantic cache tool."
    )
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["azure-cosmos", "azure-core", "openai"]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_params()
        if self.config.embedder is None:
            self._openai_client = build_openai_client(
                azure_openai_endpoint=self.config.azure_openai_endpoint,
                openai_api_key=self.config.openai_api_key,
            )
        else:
            self._openai_client = None
        self._cosmos_client = build_cosmos_client(
            cosmos_host=self.config.cosmos_host,
            key=self.config.key,
            token_credential=self.config.token_credential,
            user_agent=self.USER_AGENT,
            connection_string=self.config.connection_string,
        )
        self._owns_cosmos_client = True
        self._database = create_database_if_not_exists(
            self._cosmos_client,
            self.config.database_name,
            self.config.cosmos_database_properties,
        )
        if self.config.create_container:
            self._container = create_container_if_not_exists(
                self._database,
                self.config.container_name,
                self.config.cosmos_container_properties,
                indexing_policy=self.config.indexing_policy,
                vector_embedding_policy=self.config.vector_embedding_policy,
                full_text_policy=self.config.full_text_policy,
                default_ttl=self.config.default_ttl,
            )
        else:
            self._container = self._database.get_container_client(
                self.config.container_name
            )
        self._distance_strategy = self._infer_distance_strategy()
        self._llm_namespace = _llm_namespace(self.config.llm_string)

    def _infer_distance_strategy(self) -> DistanceStrategy:
        embeddings = (self.config.vector_embedding_policy or {}).get(
            "vectorEmbeddings"
        ) or []
        if embeddings:
            return DistanceStrategy.from_str(
                embeddings[0].get("distanceFunction", "cosine")
            )
        return DistanceStrategy.COSINE

    def _validate_params(self) -> None:
        if not self.config.create_container:
            return
        vector_indexes = (self.config.indexing_policy or {}).get("vectorIndexes")
        if not vector_indexes:
            raise ValueError(
                "vectorIndexes cannot be null or empty in the indexing_policy."
            )
        vector_embeddings = (self.config.vector_embedding_policy or {}).get(
            "vectorEmbeddings"
        )
        if not vector_embeddings:
            raise ValueError(
                "vectorEmbeddings cannot be null or empty in the "
                "vector_embedding_policy."
            )
        if self.config.cosmos_container_properties.get("partition_key") is None:
            raise ValueError("partition_key cannot be null or empty for a container.")
        if self.config.enable_hybrid_search:
            full_text_indexes = (self.config.indexing_policy or {}).get(
                "fullTextIndexes"
            )
            if not full_text_indexes:
                raise ValueError(
                    "fullTextIndexes cannot be null or empty in the indexing_policy "
                    "if hybrid search is enabled."
                )
            full_text_paths = (self.config.full_text_policy or {}).get("fullTextPaths")
            if not full_text_paths:
                raise ValueError(
                    "fullTextPaths cannot be null or empty in the full_text_policy "
                    "if hybrid search is enabled."
                )

    def _generate_embedding(self, text: str) -> list[float]:
        if self.config.embedder is not None:
            return embed_texts_via_embedder(self.config.embedder, [text])[0]
        return embed_texts(
            self._openai_client,
            [text],
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions,
        )[0]

    def _run(
        self,
        operation: str,
        prompt: str | None = None,
        response: str | None = None,
    ) -> str:
        try:
            if operation == "search":
                return self._search_cache(prompt)
            if operation == "update":
                return self._update_cache(prompt, response)
            if operation == "clear":
                return self._clear_cache()
            return json.dumps(
                {
                    "error": f"Unknown operation: {operation}",
                    "valid_operations": ["search", "update", "clear"],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Semantic cache operation failed: %s", exc)
            return json.dumps({"error": str(exc)})

    def _search_cache(self, prompt: str | None) -> str:
        if not prompt:
            return json.dumps({"error": "prompt is required for search operation"})
        try:
            query_embedding = self._generate_embedding(prompt)
            namespace_literal = quote_sql_string(self._llm_namespace)

            if self.config.enable_hybrid_search:
                # NOTE: FullTextScore / VectorDistance do not currently accept
                # parameter placeholders, so the embedding and search terms are
                # interpolated directly. FTS terms are SQL-quoted via
                # quote_sql_string to neutralise injection.
                terms = _quote_terms(prompt)
                query_sql = f"""
                SELECT TOP 1 c.id, c.prompt, c.response, c.metadata, c.llm_namespace,
                       VectorDistance(c.prompt_embedding, {query_embedding}) AS similarity_score
                FROM c
                WHERE c.llm_namespace = {namespace_literal}
                ORDER BY RANK RRF(
                    FullTextScore(c.prompt, {terms}),
                    VectorDistance(c.prompt_embedding, {query_embedding})
                )
                """  # noqa: S608
                parameters: list[dict[str, Any]] = []
            else:
                query_sql = f"""
                SELECT TOP 1 c.id, c.prompt, c.response, c.metadata, c.llm_namespace,
                       VectorDistance(c.prompt_embedding, @query_embedding) AS similarity_score
                FROM c
                WHERE c.llm_namespace = {namespace_literal}
                ORDER BY VectorDistance(c.prompt_embedding, @query_embedding)
                """  # noqa: S608
                parameters = [{"name": "@query_embedding", "value": query_embedding}]

            items = list(
                self._container.query_items(
                    query=query_sql,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )

            for item in items:
                raw_score = item.get("similarity_score", 1.0)
                # Cosmos VectorDistance semantics are distance-function-aware:
                # cosine/dotproduct return higher == better; euclidean returns
                # lower == better. score_threshold_passes encapsulates that.
                if score_threshold_passes(
                    raw_score,
                    self.config.similarity_threshold,
                    self._distance_strategy,
                ):
                    return json.dumps(
                        {
                            "cache_hit": True,
                            "similarity_score": raw_score,
                            "cached_response": item.get("response"),
                            "cache_id": item.get("id"),
                            "timestamp": item.get("metadata", {}).get("timestamp"),
                            "prompt": item.get("prompt"),
                        }
                    )

            return json.dumps(
                {
                    "cache_hit": False,
                    "similarity_score": 0.0,
                    "message": (
                        f"No cached response found above similarity threshold "
                        f"{self.config.similarity_threshold}"
                    ),
                }
            )
        except Exception as exc:
            logger.exception("Failed to search cache: %s", exc)
            return json.dumps({"error": f"Failed to search cache: {exc}"})

    def _update_cache(self, prompt: str | None, response: str | None) -> str:
        if not prompt:
            return json.dumps({"error": "prompt is required for update operation"})
        if not response:
            return json.dumps({"error": "response is required for update operation"})
        try:
            prompt_embedding = self._generate_embedding(prompt)
            partition_paths = self.config.cosmos_container_properties[
                "partition_key"
            ].get("paths", ["/agent_id"])
            # Derive a deterministic id from the namespace + prompt so that
            # re-caching the same prompt upserts the existing row instead of
            # inserting an unbounded stream of duplicate documents.
            cache_key = hashlib.sha256(
                f"{self._llm_namespace}:{prompt}".encode("utf-8")
            ).hexdigest()
            document: dict[str, Any] = {
                "id": cache_key,
                "prompt": prompt,
                "prompt_embedding": prompt_embedding,
                "response": response,
                "llm_namespace": self._llm_namespace,
                "metadata": {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "llm_string": self.config.llm_string,
                },
            }
            # Ensure each partition-key field has a default so upsert succeeds
            # when the caller has not pre-populated them on the document.
            for path in partition_paths:
                field = path.lstrip("/")
                document.setdefault(field, "default")

            if self.config.default_ttl is not None:
                document["ttl"] = self.config.default_ttl

            stored = self._container.upsert_item(body=document)
            return json.dumps(stored)
        except Exception as exc:
            logger.exception("Failed to update cache: %s", exc)
            return json.dumps({"error": f"Failed to update cache: {exc}"})

    def _clear_cache(self) -> str:
        """Delete cache entries for the current LLM namespace.

        Earlier versions of this method called ``delete_database`` which
        destroyed every container in the database — including unrelated
        memory/vector containers. We now query the matching cache rows and
        delete them individually, mirroring langchain-azure's behaviour
        (``_cache.py:CosmosDBNoSqlSemanticCache.clear``).
        """
        try:
            namespace_literal = quote_sql_string(self._llm_namespace)
            partition_paths = self.config.cosmos_container_properties[
                "partition_key"
            ].get("paths", ["/agent_id"])
            partition_field = partition_paths[0].lstrip("/")
            sql = (
                f"SELECT c.id, c.{partition_field} AS _pk FROM c "  # noqa: S608
                f"WHERE c.llm_namespace = {namespace_literal}"
            )
            items = list(
                self._container.query_items(
                    query=sql, enable_cross_partition_query=True
                )
            )
            deleted = 0
            for item in items:
                try:
                    self._container.delete_item(
                        item=item["id"], partition_key=item.get("_pk")
                    )
                    deleted += 1
                except Exception:  # noqa: PERF203
                    logger.debug(
                        "Failed to delete cache item %s",
                        item.get("id"),
                        exc_info=True,
                    )
            return json.dumps(
                {
                    "success": True,
                    "deleted_count": deleted,
                    "llm_namespace": self._llm_namespace,
                }
            )
        except Exception as exc:
            logger.exception("Failed to clear cache: %s", exc)
            return json.dumps({"error": f"Failed to clear cache: {exc}"})

    def close(self) -> None:
        """Release the underlying CosmosClient (idempotent)."""
        if getattr(self, "_owns_cosmos_client", False):
            close_cosmos_client(getattr(self, "_cosmos_client", None))
            self._owns_cosmos_client = False

    def __enter__(self) -> AzureCosmosDBSemanticCacheTool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

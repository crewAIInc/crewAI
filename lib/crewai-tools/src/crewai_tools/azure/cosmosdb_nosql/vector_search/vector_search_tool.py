"""Azure CosmosDB NoSQL vector / hybrid / full-text search tool."""

from __future__ import annotations

from collections import defaultdict
import json
from logging import getLogger
from typing import Any, ClassVar
import uuid

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
    chunked,
    quote_sql_string,
    retry_on_cosmos_throttle,
    score_threshold_passes,
    validate_sql_identifier,
)


logger = getLogger(__name__)


def _quote_terms(text: str) -> str:
    """Return a SQL-safe comma-separated list of single-quoted FTS terms."""
    return ", ".join(quote_sql_string(t) for t in text.split() if t)


class AzureCosmosDBNoSqlSearchConfig(BaseModel):
    """Per-query configuration for :class:`AzureCosmosDBNoSqlSearchTool`."""

    max_results: int | None = Field(
        default=5, description="The maximum number of items to return."
    )
    with_embedding: bool = Field(
        default=False, description="Include embedding vectors in the projection."
    )
    where: str | None = Field(
        default=None, description="Optional SQL WHERE clause appended to the query."
    )
    offset_limit: str | None = Field(
        default=None, description="Optional SQL OFFSET / LIMIT clause."
    )
    projection_mapping: dict[str, Any] | None = Field(
        default=None, description="Custom projection mapping (field -> alias)."
    )
    full_text_rank_filter: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Full text rank filters. Each item is a dict with 'search_field' and "
            "'search_text' keys, used by full_text_ranking and hybrid searches."
        ),
    )
    weights: list[float] | None = Field(
        default=None, description="Weights for hybrid RRF scoring."
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "Minimum SimilarityScore for cosine/dot-product results, or maximum "
            "distance for euclidean. None disables thresholding."
        ),
    )


class AzureCosmosDBNoSqlToolSchema(BaseModel):
    """Input schema for :class:`AzureCosmosDBNoSqlSearchTool`."""

    query: str = Field(
        ...,
        description=(
            "The search query. Pass only the query text (not a question) - it "
            "will be embedded and used to retrieve relevant documents."
        ),
    )


class AzureCosmosDBNoSqlSearchTool(BaseTool):
    """Vector / full-text / hybrid search over an Azure CosmosDB NoSQL container."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    VALID_SEARCH_TYPES: ClassVar[set[str]] = {
        "vector",
        "vector_score_threshold",
        "full_text_search",
        "full_text_ranking",
        "hybrid",
        "hybrid_score_threshold",
    }
    USER_AGENT: ClassVar[str] = "Crew-AI-CDBNoSql-VectorSearchTool-Python"

    name: str = "AzureCosmosDBNoSqlVectorSearchTool"
    description: str = (
        "Perform vector, full-text, or hybrid search over an Azure CosmosDB "
        "NoSQL container."
    )
    args_schema: type[BaseModel] = AzureCosmosDBNoSqlToolSchema

    query_config: AzureCosmosDBNoSqlSearchConfig | None = Field(
        default=None,
        description="Query-time configuration; defaults are used when omitted.",
    )
    search_type: str = Field(
        default="vector",
        description=(
            "Type of search to perform. One of: vector, vector_score_threshold, "
            "full_text_search, full_text_ranking, hybrid, hybrid_score_threshold."
        ),
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI / Azure OpenAI embedding model identifier.",
    )
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
            "Optional custom embedder satisfying the EmbedderProtocol "
            "(``embed_documents`` / ``embed_query``). Overrides OpenAI/Azure "
            "OpenAI when provided."
        ),
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        description=(
            "Azure OpenAI endpoint. If not set, the AZURE_OPENAI_ENDPOINT env "
            "var or the standard OpenAI client is used."
        ),
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI / Azure OpenAI API key (overrides env vars).",
    )
    database_name: str = Field(
        default="crewAI_database", description="CosmosDB database name."
    )
    container_name: str = Field(
        default="crewAI_container", description="CosmosDB container name."
    )
    vector_embedding_policy: dict[str, Any] | None = Field(
        default=None, description="Vector embedding policy for container creation."
    )
    indexing_policy: dict[str, Any] = Field(
        ..., description="Indexing policy for container creation."
    )
    cosmos_container_properties: dict[str, Any] = Field(
        ..., description="Properties used when creating the container."
    )
    cosmos_database_properties: dict[str, Any] = Field(
        default_factory=dict, description="Properties used when creating the database."
    )
    full_text_policy: dict[str, Any] | None = Field(
        default=None, description="Full-text policy for container creation."
    )
    text_key: str = Field(
        default="text", description="Document field that stores the source text."
    )
    embedding_key: str = Field(
        default="embedding", description="Document field that stores the embedding."
    )
    metadata_key: str = Field(
        default="metadata", description="Document field that stores metadata."
    )
    dimensions: int = Field(
        default=1536, description="Embedding vector dimensionality."
    )
    create_container: bool = Field(
        default=True,
        description="If True, create the container at init if it does not exist.",
    )
    full_text_search_enabled: bool = Field(
        default=False,
        description="Whether the container is configured for full-text search.",
    )
    table_alias: str = Field(
        default="c", description="SQL alias used for the container in queries."
    )
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["azure-cosmos", "azure-core", "openai"]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_params()
        # Identifiers spliced into SQL must be validated up front to defend
        # against accidental injection / reserved-keyword collisions.
        validate_sql_identifier(self.text_key, name="text_key")
        validate_sql_identifier(self.embedding_key, name="embedding_key")
        validate_sql_identifier(self.metadata_key, name="metadata_key")
        validate_sql_identifier(self.table_alias, name="table_alias")

        if self.embedder is None:
            self._openai_client = build_openai_client(
                azure_openai_endpoint=self.azure_openai_endpoint,
                openai_api_key=self.openai_api_key,
            )
        else:
            self._openai_client = None
        self._cosmos_client = build_cosmos_client(
            cosmos_host=self.cosmos_host,
            key=self.key,
            token_credential=self.token_credential,
            user_agent=self.USER_AGENT,
            connection_string=self.connection_string,
        )
        self._owns_cosmos_client = True
        self._database = create_database_if_not_exists(
            self._cosmos_client,
            self.database_name,
            self.cosmos_database_properties,
        )
        if self.create_container:
            self._container = create_container_if_not_exists(
                self._database,
                self.container_name,
                self.cosmos_container_properties,
                indexing_policy=self.indexing_policy,
                vector_embedding_policy=self.vector_embedding_policy,
                full_text_policy=self.full_text_policy,
            )
        else:
            self._container = self._database.get_container_client(self.container_name)
        self._distance_strategy = self._infer_distance_strategy()

    # ------------------------------------------------------------------
    # alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        *,
        indexing_policy: dict[str, Any],
        cosmos_container_properties: dict[str, Any],
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlSearchTool:
        """Construct the tool from a CosmosDB connection string."""
        return cls(
            connection_string=connection_string,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        if self.search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Invalid search_type '{self.search_type}'. "
                f"Valid options are: {sorted(self.VALID_SEARCH_TYPES)}"
            )
        if not self.create_container:
            return
        vector_indexes = (self.indexing_policy or {}).get("vectorIndexes")
        if not vector_indexes:
            raise ValueError(
                "vectorIndexes cannot be null or empty in the indexing_policy."
            )
        vector_embeddings = (self.vector_embedding_policy or {}).get("vectorEmbeddings")
        if not vector_embeddings:
            raise ValueError(
                "vectorEmbeddings cannot be null or empty in the "
                "vector_embedding_policy."
            )
        if self.cosmos_container_properties.get("partition_key") is None:
            raise ValueError("partition_key cannot be null or empty for a container.")
        if self.full_text_search_enabled:
            full_text_indexes = (self.indexing_policy or {}).get("fullTextIndexes")
            if not full_text_indexes:
                raise ValueError(
                    "fullTextIndexes cannot be null or empty in the indexing_policy "
                    "if full text search is enabled."
                )
            full_text_paths = (self.full_text_policy or {}).get("fullTextPaths")
            if not full_text_paths:
                raise ValueError(
                    "fullTextPaths cannot be null or empty in the full_text_policy "
                    "if full text search is enabled."
                )

    def _infer_distance_strategy(self) -> DistanceStrategy:
        embeddings = (self.vector_embedding_policy or {}).get("vectorEmbeddings") or []
        if embeddings:
            return DistanceStrategy.from_str(
                embeddings[0].get("distanceFunction", "cosine")
            )
        return DistanceStrategy.COSINE

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **_: Any,
    ) -> list[str]:
        """Embed ``texts`` and insert them into the container in batches."""
        if not texts:
            raise ValueError("texts cannot be null or empty")

        embeddings = self._embed_texts(texts)
        partition_key_config = self.cosmos_container_properties["partition_key"]
        if isinstance(partition_key_config, dict):
            partition_paths = partition_key_config.get("paths", ["/id"])
        elif isinstance(partition_key_config, str):
            partition_paths = [partition_key_config]
        else:
            partition_paths = ["/id"]
        # Hierarchical partition keys are resolved via the first level for
        # add_texts; callers needing finer control can pre-shape the document.
        partition_field = partition_paths[0].lstrip("/")

        meta_list = list(metadatas) if metadatas is not None else [{} for _ in texts]
        id_list = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts]

        partition_values: list[Any] = []
        for idx, meta in enumerate(meta_list):
            if partition_field == "id":
                partition_values.append(id_list[idx])
            else:
                value = meta.get(partition_field) if isinstance(meta, dict) else None
                if value is None:
                    raise ValueError(
                        f"Partition key '{partition_field}' not found in metadata "
                        f"at index {idx}"
                    )
                partition_values.append(value)

        items = [
            {
                "id": item_id,
                "pk": pk,
                self.text_key: text,
                self.embedding_key: embedding,
                self.metadata_key: meta,
            }
            for item_id, pk, text, meta, embedding in zip(
                id_list, partition_values, texts, meta_list, embeddings, strict=False
            )
        ]

        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            grouped[item["pk"]].append(item)

        batch_size = 25
        doc_ids: list[str] = []
        for pk_value, group in grouped.items():
            for batch in chunked(group, batch_size):
                self._batch_create(batch, pk_value)
                doc_ids.extend(doc["id"] for doc in batch)
        return doc_ids

    @retry_on_cosmos_throttle()
    def _batch_create(self, batch: list[dict[str, Any]], pk_value: Any) -> None:
        # execute_item_batch expects each operation as (type, args_tuple[, kwargs]);
        # the create body must be wrapped in a 1-tuple, i.e. ("create", (doc,)).
        self._container.execute_item_batch(
            batch_operations=[("create", (doc,)) for doc in batch],
            partition_key=pk_value,
        )

    def delete_by_id(self, document_id: str, partition_key_value: Any) -> bool:
        """Delete a single document; returns True on success, False on miss."""
        try:
            self._container.delete_item(
                item=document_id, partition_key=partition_key_value
            )
            return True
        except Exception as exc:  # pragma: no cover - depends on SDK
            logger.warning(
                "delete_by_id failed for %s/%s: %s",
                document_id,
                partition_key_value,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # embedding
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is not None:
            return embed_texts_via_embedder(self.embedder, texts)
        return embed_texts(
            self._openai_client,
            texts,
            model=self.embedding_model,
            dimensions=self.dimensions,
        )

    # ------------------------------------------------------------------
    # tool entry point
    # ------------------------------------------------------------------

    def _run(self, query: str) -> str:
        try:
            search_query, parameters = self._construct_query(query=query)
            results = self._execute_query(search_query, parameters)
            return json.dumps(results)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Vector search failed: %s", exc)
            return json.dumps({"error": str(exc)})

    def _normalised_search_type(self) -> str:
        """Strip _score_threshold suffix; keeps SQL generation simple."""
        if self.search_type == "vector_score_threshold":
            return "vector"
        if self.search_type == "hybrid_score_threshold":
            return "hybrid"
        return self.search_type

    def _construct_query(self, query: str) -> tuple[str, list[dict[str, Any]]]:
        config = self.query_config or AzureCosmosDBNoSqlSearchConfig()
        effective_type = self._normalised_search_type()
        # Only vector / hybrid searches need an embedding; pure full-text
        # searches must not force an embedder (or an OpenAI key) to be present.
        embeddings = (
            self._embed_texts([query])[0]
            if effective_type in ("vector", "hybrid")
            else None
        )

        if effective_type in ("full_text_ranking", "hybrid"):
            sql = f"SELECT {'TOP ' + str(config.max_results) + ' ' if not config.offset_limit else ''}"
        else:
            sql = f"""SELECT {"TOP @top " if not config.offset_limit else ""}"""

        sql += self._generate_projection_fields(embeddings, effective_type)
        table = self.table_alias
        sql += f" FROM {table}"

        if config.where:
            sql += f" WHERE {config.where}"

        # NOTE: Cosmos NoSQL VectorDistance/FullTextScore do not currently
        # accept @param placeholders, so embedding values and FTS terms must
        # be interpolated. Embedding values come from the embedder; FTS terms
        # are SQL-quoted via quote_sql_string to defend against injection.
        if effective_type == "full_text_ranking":
            if config.full_text_rank_filter is None:
                raise ValueError(
                    "full_text_rank_filter cannot be None for full_text_ranking queries."
                )
            if len(config.full_text_rank_filter) == 1:
                f = config.full_text_rank_filter[0]
                field = validate_sql_identifier(f["search_field"], name="search_field")
                terms = _quote_terms(f["search_text"])
                sql += f" ORDER BY RANK FullTextScore({table}.{field}, {terms})"
            else:
                rank_components = [
                    f"FullTextScore({table}."
                    f"{validate_sql_identifier(item['search_field'], name='search_field')}, "
                    f"{_quote_terms(item['search_text'])})"
                    for item in config.full_text_rank_filter
                ]
                sql += f" ORDER BY RANK RRF({', '.join(rank_components)})"
        elif effective_type == "vector":
            sql += f" ORDER BY VectorDistance({table}[@embeddingKey], @embeddings)"
        elif effective_type == "hybrid":
            if config.full_text_rank_filter is None:
                raise ValueError(
                    "full_text_rank_filter cannot be None for hybrid queries."
                )
            rank_components = [
                f"FullTextScore({table}."
                f"{validate_sql_identifier(item['search_field'], name='search_field')}, "
                f"{_quote_terms(item['search_text'])})"
                for item in config.full_text_rank_filter
            ]
            sql += (
                f" ORDER BY RANK RRF({', '.join(rank_components)}, "
                f"VectorDistance({table}.{self.embedding_key}, {embeddings})"
            )
            if config.weights:
                sql += f", {config.weights})"
            else:
                sql += ")"

        if config.offset_limit is not None:
            sql += f" {config.offset_limit}"

        parameters: list[dict[str, Any]] = []
        if effective_type in ("full_text_search", "vector"):
            parameters = self._build_parameters(embeddings)
        return sql, parameters

    def _generate_projection_fields(
        self,
        embeddings: list[float] | None = None,
        effective_type: str | None = None,
    ) -> str:
        config = self.query_config or AzureCosmosDBNoSqlSearchConfig()
        table = self.table_alias
        effective_type = effective_type or self._normalised_search_type()

        if effective_type in ("full_text_ranking", "hybrid"):
            if config.projection_mapping:
                projection = ", ".join(
                    f"{table}.{validate_sql_identifier(key, name='projection key')} "
                    f"as {validate_sql_identifier(alias, name='projection alias')}"
                    for key, alias in config.projection_mapping.items()
                )
            elif config.full_text_rank_filter:
                projection = f"{table}.id, " + ", ".join(
                    f"{table}.{validate_sql_identifier(item['search_field'], name='search_field')} "
                    f"as {validate_sql_identifier(item['search_field'], name='search_field')}"
                    for item in config.full_text_rank_filter
                )
            else:
                projection = (
                    f"{table}.id, {table}.{self.text_key} as description, "
                    f"{table}.{self.metadata_key} as metadata"
                )
            if effective_type == "hybrid":
                if config.with_embedding:
                    projection += f", {table}.{self.embedding_key} as embedding"
                projection += (
                    f", VectorDistance({table}.{self.embedding_key}, "
                    f"{embeddings}) as SimilarityScore"
                )
            return projection

        if config.projection_mapping:
            projection = ", ".join(
                f"{table}[@{validate_sql_identifier(key, name='projection key')}] "
                f"as {validate_sql_identifier(alias, name='projection alias')}"
                for key, alias in config.projection_mapping.items()
            )
        elif config.full_text_rank_filter:
            projection = f"{table}.id, " + ", ".join(
                f"{table}.{validate_sql_identifier(item['search_field'], name='search_field')} "
                f"as {validate_sql_identifier(item['search_field'], name='search_field')}"
                for item in config.full_text_rank_filter
            )
        else:
            projection = (
                f"{table}.id, {table}[@textKey] as description, "
                f"{table}[@metadataKey] as metadata"
            )

        if effective_type == "vector":
            if config.with_embedding:
                projection += f", {table}[@embeddingKey] as embedding"
            projection += (
                f", VectorDistance({table}[@embeddingKey], "
                "@embeddings) as SimilarityScore"
            )
        return projection

    def _build_parameters(self, embeddings: list[float] | None) -> list[dict[str, Any]]:
        config = self.query_config or AzureCosmosDBNoSqlSearchConfig()
        parameters: list[dict[str, Any]] = [
            {"name": "@top", "value": config.max_results},
        ]

        if config.projection_mapping:
            for key in config.projection_mapping:
                parameters.append({"name": f"@{key}", "value": key})  # noqa: PERF401
        else:
            parameters.append({"name": "@textKey", "value": self.text_key})
            parameters.append({"name": "@metadataKey", "value": self.metadata_key})

        if self._normalised_search_type() == "vector":
            parameters.append({"name": "@embeddingKey", "value": self.embedding_key})
            parameters.append({"name": "@embeddings", "value": embeddings})
        return parameters

    @retry_on_cosmos_throttle()
    def _query_items(
        self, query: str, parameters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return list(
            self._container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            )
        )

    def _execute_query(
        self, query: str, parameters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        config = self.query_config or AzureCosmosDBNoSqlSearchConfig()
        items = self._query_items(query, parameters)
        results: list[dict[str, Any]] = []
        threshold_search = self.search_type in (
            "vector_score_threshold",
            "hybrid_score_threshold",
        )
        for item in items:
            if self._normalised_search_type() in ("vector", "hybrid"):
                score = item.get("SimilarityScore", 0.0)
                # For the *_score_threshold types, always apply the configured
                # threshold (None => keep all). For the plain "vector"/"hybrid"
                # types, only filter when the caller explicitly set a threshold;
                # applying a 0.0 default here would wrongly drop every result
                # for euclidean distance (where lower == more similar).
                if threshold_search:
                    if not score_threshold_passes(
                        score, config.threshold, self._distance_strategy
                    ):
                        continue
                elif config.threshold is not None:
                    if not score_threshold_passes(
                        score, config.threshold, self._distance_strategy
                    ):
                        continue
            results.append(item)
        return results

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying CosmosClient (idempotent)."""
        if getattr(self, "_owns_cosmos_client", False):
            close_cosmos_client(getattr(self, "_cosmos_client", None))
            self._owns_cosmos_client = False

    def __enter__(self) -> AzureCosmosDBNoSqlSearchTool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

"""Azure CosmosDB NoSQL agent-memory CRUD tool."""

from __future__ import annotations

from datetime import datetime, timezone
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
from crewai_tools.azure.cosmosdb_nosql._utils import (
    quote_sql_string,
    validate_sql_identifier,
)


logger = getLogger(__name__)


class AzureCosmosDBMemoryConfig(BaseModel):
    """Configuration for :class:`AzureCosmosDBMemoryTool`."""

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
    use_optimistic_concurrency: bool = Field(
        default=False,
        description=(
            "If True, ``update`` uses the document's _etag with "
            "MatchConditions.IfNotModified to detect concurrent modifications "
            "and surface a 412 PreconditionFailed error."
        ),
    )


class AzureCosmosDBMemoryToolSchema(BaseModel):
    """Input schema for :class:`AzureCosmosDBMemoryTool`."""

    operation: str = Field(
        ...,
        description=(
            "Operation: 'store', 'read', 'retrieve', 'update', 'delete', or 'clear'."
        ),
    )
    memory_item: dict[str, Any] | None = Field(
        default=None,
        description="Memory document for 'store' and 'update' operations.",
    )
    partition_key_value: str | list[str] | None = Field(
        default=None,
        description=(
            "Partition key value. Provide a list for hierarchical partition keys."
        ),
    )
    memory_id: str | None = Field(
        default=None,
        description="Document id for 'read', 'update' and 'delete' operations.",
    )
    query_filter: dict[str, Any] | None = Field(
        default=None,
        description="Equality filters applied to 'c.content' fields on 'retrieve'.",
    )
    max_results: int | None = Field(
        default=10, description="Maximum number of items returned by 'retrieve'."
    )
    ttl: int | None = Field(
        default=None, description="Time-to-live in seconds for the stored document."
    )


class AzureCosmosDBMemoryTool(BaseTool):
    """CRUD operations on agent memory items stored in Azure CosmosDB NoSQL."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    USER_AGENT: ClassVar[str] = "CrewAI-CosmosDB-Memory-Tool-Python"

    name: str = "AzureCosmosDBMemoryTool"
    description: str = (
        "Store, retrieve, update, and delete memory items in Azure CosmosDB."
    )
    args_schema: type[BaseModel] = AzureCosmosDBMemoryToolSchema

    config: AzureCosmosDBMemoryConfig = Field(
        ..., description="Configuration for the memory tool."
    )
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["azure-cosmos", "azure-core"]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
            )
        else:
            self._container = self._database.get_container_client(
                self.config.container_name
            )

    @staticmethod
    def _format_partition_value(value: Any) -> str:
        """Quote a partition value for safe SQL interpolation."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return quote_sql_string(value)

    def _get_partition_key_fields(self) -> tuple[list[str], Any]:
        partition_key_config = self.config.cosmos_container_properties.get(
            "partition_key", {}
        )
        if isinstance(partition_key_config, dict):
            paths = partition_key_config.get("paths", ["/agent_id"])
        elif isinstance(partition_key_config, str):
            paths = [partition_key_config]
        else:
            paths = ["/agent_id"]
        field_names = [path.lstrip("/") for path in paths]
        return field_names, partition_key_config

    @classmethod
    def _build_partition_key_filter(
        cls,
        partition_key_value: str | list[str],
        field_names: list[str],
    ) -> str:
        for field in field_names:
            validate_sql_identifier(field, name="partition_key field")
        if isinstance(partition_key_value, str):
            if len(field_names) > 1:
                raise ValueError(
                    f"Container has hierarchical partition key with "
                    f"{len(field_names)} levels, but only one value provided"
                )
            return f"c.{field_names[0]} = {cls._format_partition_value(partition_key_value)}"
        if len(partition_key_value) != len(field_names):
            raise ValueError(
                f"Container has {len(field_names)} partition key levels, but "
                f"{len(partition_key_value)} values provided"
            )
        return " AND ".join(
            f"c.{field} = {cls._format_partition_value(value)}"
            for field, value in zip(field_names, partition_key_value, strict=False)
        )

    def _run(
        self,
        operation: str,
        memory_item: dict[str, Any] | None = None,
        partition_key_value: str | list[str] | None = None,
        memory_id: str | None = None,
        query_filter: dict[str, Any] | None = None,
        max_results: int | None = 10,
        ttl: int | None = None,
    ) -> str:
        try:
            if operation == "store":
                if memory_item is None:
                    return json.dumps(
                        {"error": "memory_item is required for store operation"}
                    )
                return self._store_memory(memory_item, ttl)
            if operation == "read":
                return self._read_memory(partition_key_value, memory_id)
            if operation == "retrieve":
                return self._retrieve_memory(
                    partition_key_value, query_filter, max_results
                )
            if operation == "update":
                if memory_item is None:
                    return json.dumps(
                        {"error": "memory_item is required for update operation"}
                    )
                return self._update_memory(
                    partition_key_value, memory_id, memory_item, ttl
                )
            if operation == "delete":
                return self._delete_memory(partition_key_value, memory_id)
            if operation == "clear":
                return self._clear_memory(partition_key_value)
            return json.dumps(
                {
                    "error": f"Unknown operation: {operation}",
                    "valid_operations": [
                        "store",
                        "read",
                        "retrieve",
                        "update",
                        "delete",
                        "clear",
                    ],
                }
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Memory operation failed: %s", exc)
            return json.dumps({"error": str(exc)})

    def _store_memory(self, memory_item: dict[str, Any], ttl: int | None = None) -> str:
        if ttl is not None:
            memory_item["ttl"] = ttl
        memory_item.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        try:
            response = self._container.create_item(body=memory_item)
            return json.dumps(response)
        except Exception as exc:
            logger.exception("Failed to store memory: %s", exc)
            return json.dumps({"error": f"Failed to store memory: {exc}"})

    def _read_memory(
        self,
        partition_key_value: str | list[str] | None,
        memory_id: str | None,
    ) -> str:
        if not partition_key_value:
            return json.dumps(
                {"error": "partition_key_value is required for read operation"}
            )
        if not memory_id:
            return json.dumps({"error": "memory_id is required for read operation"})
        try:
            response = self._container.read_item(
                item=memory_id, partition_key=partition_key_value
            )
            return json.dumps(response)
        except Exception as exc:
            logger.exception("Failed to read memory: %s", exc)
            return json.dumps({"error": f"Failed to read memory: {exc}"})

    def _retrieve_memory(
        self,
        partition_key_value: str | list[str] | None,
        query_filter: dict[str, Any] | None = None,
        max_results: int | None = 10,
    ) -> str:
        if not partition_key_value:
            return json.dumps(
                {"error": "partition_key_value is required for retrieve operation"}
            )
        try:
            field_names, _ = self._get_partition_key_fields()
            partition_filter = self._build_partition_key_filter(
                partition_key_value, field_names
            )
            # Only emit TOP for a positive limit; a negative/zero value would
            # produce "TOP -1" which Cosmos rejects.
            top_clause = (
                f"TOP {int(max_results)} "
                if max_results is not None and int(max_results) > 0
                else ""
            )
            query_sql = f"SELECT {top_clause}* FROM c WHERE {partition_filter}"  # noqa: S608

            if query_filter:
                for key, value in query_filter.items():
                    validate_sql_identifier(key, name="query_filter key")
                    query_sql += (
                        f" AND c.content.{key} = {self._format_partition_value(value)}"
                    )

            query_sql += " ORDER BY c.created_at DESC"

            items = list(
                self._container.query_items(
                    query=query_sql,
                    partition_key=partition_key_value,
                    enable_cross_partition_query=False,
                )
            )
            return json.dumps(items)
        except Exception as exc:
            logger.exception("Failed to retrieve memory: %s", exc)
            return json.dumps({"error": f"Failed to retrieve memory: {exc}"})

    def _update_memory(
        self,
        partition_key_value: str | list[str] | None,
        memory_id: str | None,
        upsert_item: dict[str, Any],
        ttl: int | None = None,
    ) -> str:
        if not partition_key_value:
            return json.dumps(
                {"error": "partition_key_value is required for update operation"}
            )
        if not memory_id:
            return json.dumps({"error": "memory_id is required for update operation"})
        try:
            existing = self._container.read_item(
                item=memory_id, partition_key=partition_key_value
            )
            # Merge onto the existing document so fields the caller did not
            # resend (e.g. created_at, a prior ttl) are preserved rather than
            # dropped by replace_item's whole-document semantics.
            merged = {**existing, **upsert_item}
            merged["id"] = memory_id
            if ttl is not None:
                merged["ttl"] = ttl

            if self.config.use_optimistic_concurrency:
                # Use ETag from the just-read document so a concurrent writer
                # who has modified the row in between will be detected.
                from azure.core import MatchConditions  # local import; optional dep

                etag = existing.get("_etag")
                response = self._container.replace_item(
                    item=memory_id,
                    body=merged,
                    etag=etag,
                    match_condition=MatchConditions.IfNotModified,
                )
            else:
                response = self._container.replace_item(item=memory_id, body=merged)
            return json.dumps(response)
        except Exception as exc:
            logger.exception("Failed to update memory: %s", exc)
            return json.dumps({"error": f"Failed to update memory: {exc}"})

    def _delete_memory(
        self,
        partition_key_value: str | list[str] | None,
        memory_id: str | None,
    ) -> str:
        if not partition_key_value:
            return json.dumps(
                {"error": "partition_key_value is required for delete operation"}
            )
        if not memory_id:
            return json.dumps({"error": "memory_id is required for delete operation"})
        try:
            self._container.delete_item(
                item=memory_id, partition_key=partition_key_value
            )
            return json.dumps(
                {"success": True, "message": f"Item {memory_id} has been deleted"}
            )
        except Exception as exc:
            logger.exception("Failed to delete memory: %s", exc)
            return json.dumps({"error": f"Failed to delete memory: {exc}"})

    def _clear_memory(self, partition_key_value: str | list[str] | None) -> str:
        if not partition_key_value:
            return json.dumps(
                {"error": "partition_key_value is required for clear operation"}
            )
        field_names, _ = self._get_partition_key_fields()
        partition_filter = self._build_partition_key_filter(
            partition_key_value, field_names
        )
        query_sql = f"SELECT c.id FROM c WHERE {partition_filter}"  # noqa: S608
        # Track the count outside the try so a mid-loop batch failure still
        # reports how many documents were already (transactionally) deleted.
        deleted_count = 0
        batch_size = 100
        try:
            # Stream ids and delete in fixed-size batches instead of loading the
            # whole (up to 20 GB) logical partition into memory at once.
            pending: list[str] = []
            for item in self._container.query_items(
                query=query_sql,
                partition_key=partition_key_value,
                enable_cross_partition_query=False,
            ):
                pending.append(item["id"])
                if len(pending) >= batch_size:
                    self._container.execute_item_batch(
                        batch_operations=[("delete", (doc_id,)) for doc_id in pending],
                        partition_key=partition_key_value,
                    )
                    deleted_count += len(pending)
                    pending = []
            if pending:
                self._container.execute_item_batch(
                    batch_operations=[("delete", (doc_id,)) for doc_id in pending],
                    partition_key=partition_key_value,
                )
                deleted_count += len(pending)

            return json.dumps(
                {
                    "success": True,
                    "partition_key_value": partition_key_value,
                    "operation": "clear",
                    "deleted_count": deleted_count,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        except Exception as exc:
            logger.exception("Failed to clear memory: %s", exc)
            return json.dumps(
                {
                    "error": f"Failed to clear memory: {exc}",
                    "deleted_count": deleted_count,
                }
            )

    def close(self) -> None:
        """Release the underlying CosmosClient (idempotent)."""
        if getattr(self, "_owns_cosmos_client", False):
            close_cosmos_client(getattr(self, "_cosmos_client", None))
            self._owns_cosmos_client = False

    def __enter__(self) -> AzureCosmosDBMemoryTool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

"""Shared Azure CosmosDB NoSQL client helpers.

These helpers centralise the client/database/container construction logic that
the three CosmosDB tools (vector search, semantic cache, memory store) all
share. They also gate the optional ``azure-cosmos`` / ``azure-identity``
dependency behind a clean lazy import so that simply importing
``crewai_tools`` does not require the extra to be installed.
"""

from __future__ import annotations

from typing import Any


_INSTALL_HINT = (
    "Azure CosmosDB tools require the optional 'azure-cosmosdb' extra. "
    "Install it with: pip install 'crewai-tools[azure-cosmosdb]'"
)


def require_azure_cosmos() -> Any:
    """Import and return the ``azure.cosmos`` module, with a friendly error."""
    try:
        import azure.cosmos  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise ImportError(_INSTALL_HINT) from exc
    import azure.cosmos as azure_cosmos

    return azure_cosmos


def build_cosmos_client(
    cosmos_host: str | None = None,
    key: str | None = None,
    token_credential: Any | None = None,
    user_agent: str = "Crew-AI-CDBNoSql-Python",
    *,
    connection_string: str | None = None,
) -> Any:
    """Construct a CosmosClient from a key, credential or connection string.

    Exactly one auth source must be supplied. ``connection_string`` is
    preferred when set; otherwise either ``key`` (with ``cosmos_host``) or
    ``token_credential`` (with ``cosmos_host``) is used.
    """
    azure_cosmos = require_azure_cosmos()

    sources = sum(
        1 for src in (key, token_credential, connection_string) if src is not None
    )
    if sources == 0:
        raise ValueError(
            "Provide one of 'key', 'token_credential' or 'connection_string'."
        )
    if sources > 1:
        raise ValueError(
            "Provide only one of 'key', 'token_credential' or 'connection_string'."
        )

    if connection_string is not None:
        return azure_cosmos.CosmosClient.from_connection_string(
            connection_string, user_agent=user_agent
        )
    if not cosmos_host:
        raise ValueError(
            "'cosmos_host' is required when authenticating with a key or token "
            "credential."
        )
    if key is not None:
        return azure_cosmos.CosmosClient(cosmos_host, key, user_agent=user_agent)
    return azure_cosmos.CosmosClient(
        cosmos_host, token_credential, user_agent=user_agent
    )


def close_cosmos_client(client: Any) -> None:
    """Best-effort close of a CosmosClient; safe to call multiple times."""
    if client is None:
        return
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:  # pragma: no cover - defensive cleanup  # noqa: S110
        pass


def create_database_if_not_exists(
    cosmos_client: Any,
    database_name: str,
    database_properties: dict[str, Any] | None = None,
) -> Any:
    """Create the database if it does not already exist."""
    props = database_properties or {}
    return cosmos_client.create_database_if_not_exists(
        id=database_name,
        offer_throughput=props.get("offer_throughput"),
        session_token=props.get("session_token"),
        initial_headers=props.get("initial_headers"),
        etag=props.get("etag"),
        match_condition=props.get("match_condition"),
    )


def create_container_if_not_exists(
    database: Any,
    container_name: str,
    container_properties: dict[str, Any],
    indexing_policy: dict[str, Any] | None = None,
    vector_embedding_policy: dict[str, Any] | None = None,
    full_text_policy: dict[str, Any] | None = None,
    default_ttl: int | None = None,
) -> Any:
    """Create the container if it does not already exist."""
    return database.create_container_if_not_exists(
        id=container_name,
        partition_key=container_properties["partition_key"],
        indexing_policy=indexing_policy,
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
        default_ttl=default_ttl
        if default_ttl is not None
        else container_properties.get("default_ttl"),
        offer_throughput=container_properties.get("offer_throughput"),
        unique_key_policy=container_properties.get("unique_key_policy"),
        conflict_resolution_policy=container_properties.get(
            "conflict_resolution_policy"
        ),
        analytical_storage_ttl=container_properties.get("analytical_storage_ttl"),
        computed_properties=container_properties.get("computed_properties"),
        etag=container_properties.get("etag"),
        match_condition=container_properties.get("match_condition"),
        session_token=container_properties.get("session_token"),
        initial_headers=container_properties.get("initial_headers"),
    )

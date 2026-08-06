"""Tests for the CosmosDB NoSQL memory storage backend.

The Azure SDK is mocked end-to-end so these tests run with no live Cosmos
account and without requiring the ``cosmosdb`` extra to be installed.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# azure.cosmos stub (installed before the module under test is imported)
# ---------------------------------------------------------------------------


def _install_azure_cosmos_stub() -> None:
    if "azure.cosmos" in sys.modules:
        return

    azure_pkg = sys.modules.setdefault("azure", types.ModuleType("azure"))
    cosmos_module = types.ModuleType("azure.cosmos")

    class _PartitionKey:
        def __init__(self, path: str, kind: str = "Hash") -> None:
            self.path = path
            self.kind = kind

    class _CosmosClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def create_database_if_not_exists(self, **_: Any) -> Any:
            return MagicMock(name="database")

    cosmos_module.CosmosClient = _CosmosClient
    cosmos_module.PartitionKey = _PartitionKey
    cosmos_module.exceptions = types.SimpleNamespace(
        CosmosResourceNotFoundError=type("CosmosResourceNotFoundError", (Exception,), {})
    )
    sys.modules["azure.cosmos"] = cosmos_module
    setattr(azure_pkg, "cosmos", cosmos_module)


_install_azure_cosmos_stub()

from crewai.memory.storage.cosmosdb_nosql_storage import (  # noqa: E402
    CosmosDBNoSqlStorage,
    _score_from_distance,
)
from crewai.memory.types import MemoryRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_container() -> MagicMock:
    return MagicMock(name="container")


@pytest.fixture
def storage(mock_container: MagicMock) -> CosmosDBNoSqlStorage:
    """Build a storage backend whose Cosmos plumbing is fully mocked."""
    db = MagicMock(name="database")
    db.create_container_if_not_exists.return_value = mock_container
    db.get_container_client.return_value = mock_container

    client = MagicMock(name="client")
    client.create_database_if_not_exists.return_value = db

    with patch(
        "crewai.memory.storage.cosmosdb_nosql_storage._require_azure_cosmos"
    ) as require:
        azure_cosmos = MagicMock()
        azure_cosmos.CosmosClient.return_value = client
        azure_cosmos.PartitionKey.side_effect = lambda **kw: kw
        require.return_value = azure_cosmos

        store = CosmosDBNoSqlStorage(
            cosmos_host="https://example.documents.azure.com:443/",
            key="fake-key",
            vector_dim=4,
        )
    store._container = mock_container  # ensure tests target the right mock
    return store


def _record(content: str = "hello", scope: str = "/", **overrides: Any) -> MemoryRecord:
    payload: dict[str, Any] = dict(
        content=content,
        scope=scope,
        categories=[],
        importance=0.5,
        embedding=[0.1, 0.2, 0.3, 0.4],
        metadata={},
    )
    payload.update(overrides)
    return MemoryRecord(**payload)


# ---------------------------------------------------------------------------
# Lazy-import / install hint
# ---------------------------------------------------------------------------


def test_lazy_import_raises_with_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """If azure-cosmos is missing, the error must mention the extras name."""
    from crewai.memory.storage import cosmosdb_nosql_storage as mod

    # Force the import inside _require_azure_cosmos to fail by hiding the
    # already-loaded stub from sys.modules and short-circuiting the loader.
    monkeypatch.delitem(sys.modules, "azure.cosmos", raising=False)
    monkeypatch.delitem(sys.modules, "azure", raising=False)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "azure.cosmos" or name.startswith("azure.cosmos."):
            raise ImportError("simulated missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    with pytest.raises(ImportError) as exc:
        mod._require_azure_cosmos()
    assert "cosmosdb" in str(exc.value)


# ---------------------------------------------------------------------------
# Distance scoring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,fn,expected_high_better",
    [
        (0.95, "cosine", True),
        (0.95, "dotProduct", True),
        (0.05, "euclidean", True),  # transformed: 1/(1+0.05) ~= 0.952
    ],
)
def test_score_from_distance_orientation(
    raw: float, fn: str, expected_high_better: bool
) -> None:
    """Score must be 'higher = better' regardless of distance function."""
    s = _score_from_distance(raw, fn)
    assert 0.0 <= s <= 1.0
    if expected_high_better:
        assert s > 0.5


def test_score_from_distance_euclidean_smaller_distance_higher_score() -> None:
    near = _score_from_distance(0.01, "euclidean")
    far = _score_from_distance(2.0, "euclidean")
    assert near > far


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_save_upserts_each_record(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    storage.save([_record(content="a"), _record(content="b")])
    assert mock_container.upsert_item.call_count == 2


def test_get_record_returns_none_when_missing(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    mock_container.query_items.return_value = iter([])
    assert storage.get_record("missing-id") is None


def test_get_record_round_trip(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    mock_container.query_items.return_value = iter(
        [
            {
                "id": "rec-1",
                "scope": "/agents/a1",
                "content": "remembered",
                "categories": ["k"],
                "metadata": {"source": "test"},
                "importance": 0.7,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": None,
                "source": None,
                "private": False,
                "embedding": [0.1, 0.2, 0.3, 0.4],
            }
        ]
    )
    rec = storage.get_record("rec-1")
    assert rec is not None
    assert rec.content == "remembered"
    assert rec.scope == "/agents/a1"


def test_search_uses_distance_aware_score_for_cosine(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    """For cosine, Cosmos VectorDistance is similarity-like (higher = better)."""
    mock_container.query_items.return_value = iter(
        [
            {
                "id": "doc-good",
                "scope": "/",
                "content": "good",
                "categories": [],
                "metadata": {},
                "importance": 0.5,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": None,
                "source": None,
                "private": False,
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "_distance": 0.92,  # similarity-like (cosine)
            },
            {
                "id": "doc-bad",
                "scope": "/",
                "content": "bad",
                "categories": [],
                "metadata": {},
                "importance": 0.5,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_accessed": None,
                "source": None,
                "private": False,
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "_distance": 0.10,
            },
        ]
    )
    results = storage.search([0.1, 0.2, 0.3, 0.4], min_score=0.5)
    assert [r.content for r, _ in results] == ["good"]
    # The kept score is the raw similarity for cosine.
    assert results[0][1] == pytest.approx(0.92)


def test_search_empty_query_returns_empty(storage: CosmosDBNoSqlStorage) -> None:
    assert storage.search([]) == []


# ---------------------------------------------------------------------------
# Filter SQL safety
# ---------------------------------------------------------------------------


def test_metadata_filter_rejects_non_alphanumeric_key() -> None:
    with pytest.raises(ValueError):
        CosmosDBNoSqlStorage._metadata_filter({"bad key": "x"})


def test_metadata_filter_quotes_string_values_safely() -> None:
    sql = CosmosDBNoSqlStorage._metadata_filter({"author": "O'Reilly"})
    assert sql is not None
    # Single quotes inside the value must be doubled per SQL escape rules.
    assert "O''Reilly" in sql


def test_scope_prefix_filter_quotes_value() -> None:
    sql = CosmosDBNoSqlStorage._scope_prefix_filter("/agents/a'1")
    assert sql is not None
    assert "a''1" in sql


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


def test_asave_dispatches_to_save(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    asyncio.run(storage.asave([_record()]))
    assert mock_container.upsert_item.call_count == 1


# ---------------------------------------------------------------------------
# Auth: connection string + from_env()
# ---------------------------------------------------------------------------


def _mock_azure_cosmos() -> MagicMock:
    db = MagicMock(name="database")
    container = MagicMock(name="container")
    db.create_container_if_not_exists.return_value = container
    db.get_container_client.return_value = container
    client = MagicMock(name="client")
    client.create_database_if_not_exists.return_value = db

    azure_cosmos = MagicMock()
    azure_cosmos.CosmosClient.return_value = client
    azure_cosmos.CosmosClient.from_connection_string.return_value = client
    azure_cosmos.PartitionKey.side_effect = lambda **kw: kw
    return azure_cosmos


def test_init_connection_string_conflicts_with_key() -> None:
    azure_cosmos = _mock_azure_cosmos()
    with patch(
        "crewai.memory.storage.cosmosdb_nosql_storage._require_azure_cosmos",
        return_value=azure_cosmos,
    ):
        with pytest.raises(ValueError, match="connection_string"):
            CosmosDBNoSqlStorage(connection_string="conn", key="k", vector_dim=4)


def test_init_connection_string_builds_client_from_connection_string() -> None:
    azure_cosmos = _mock_azure_cosmos()
    with patch(
        "crewai.memory.storage.cosmosdb_nosql_storage._require_azure_cosmos",
        return_value=azure_cosmos,
    ):
        CosmosDBNoSqlStorage(connection_string="AccountEndpoint=x;AccountKey=y;", vector_dim=4)
    azure_cosmos.CosmosClient.from_connection_string.assert_called_once()
    azure_cosmos.CosmosClient.assert_not_called()


def test_from_env_prefers_connection_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_COSMOS_CONNECTION_STRING", "AccountEndpoint=x;AccountKey=y;")
    monkeypatch.delenv("AZURE_COSMOS_HOST", raising=False)
    monkeypatch.delenv("AZURE_COSMOS_KEY", raising=False)
    azure_cosmos = _mock_azure_cosmos()
    with patch(
        "crewai.memory.storage.cosmosdb_nosql_storage._require_azure_cosmos",
        return_value=azure_cosmos,
    ):
        store = CosmosDBNoSqlStorage.from_env(vector_dim=4)
    assert store is not None
    azure_cosmos.CosmosClient.from_connection_string.assert_called_once()


def test_from_env_host_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_COSMOS_CONNECTION_STRING", raising=False)
    monkeypatch.setenv("AZURE_COSMOS_HOST", "https://acct.documents.azure.com:443/")
    monkeypatch.setenv("AZURE_COSMOS_KEY", "fake-key")
    monkeypatch.setenv("AZURE_COSMOS_DATABASE_NAME", "envdb")
    azure_cosmos = _mock_azure_cosmos()
    with patch(
        "crewai.memory.storage.cosmosdb_nosql_storage._require_azure_cosmos",
        return_value=azure_cosmos,
    ):
        store = CosmosDBNoSqlStorage.from_env(vector_dim=4)
    call = azure_cosmos.CosmosClient.call_args
    assert call.args[0] == "https://acct.documents.azure.com:443/"
    assert call.args[1] == "fake-key"
    assert store._database_name == "envdb"


def test_from_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AZURE_COSMOS_CONNECTION_STRING",
        "AZURE_COSMOS_HOST",
        "AZURE_COSMOS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ValueError, match="from_env"):
        CosmosDBNoSqlStorage.from_env()


def test_memory_selector_cosmosdb_uses_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory(storage='cosmosdb') must route to CosmosDBNoSqlStorage.from_env()."""
    from crewai.memory.unified_memory import Memory

    sentinel = MagicMock(name="cosmos-storage")
    with patch.object(
        CosmosDBNoSqlStorage, "from_env", return_value=sentinel
    ) as from_env:
        mem = Memory(storage="cosmosdb", llm=MagicMock(), embedder=MagicMock())
    from_env.assert_called_once()
    assert mem._storage is sentinel


# ---------------------------------------------------------------------------
# Review-fix regressions
# ---------------------------------------------------------------------------


def test_parse_dt_normalizes_aware_to_naive_utc() -> None:
    from crewai.memory.storage.cosmosdb_nosql_storage import _parse_dt

    dt = _parse_dt("2024-01-01T12:00:00+02:00")
    assert dt.tzinfo is None
    assert (dt.hour, dt.minute) == (10, 0)  # converted to UTC


def test_record_to_doc_stores_naive_utc_iso(storage: CosmosDBNoSqlStorage) -> None:
    rec = _record()
    rec.created_at = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    doc = storage._record_to_doc(rec)
    assert "+" not in doc["created_at"] and "Z" not in doc["created_at"]


def test_close_releases_client(storage: CosmosDBNoSqlStorage) -> None:
    client = storage._client
    storage.close()
    client.close.assert_called_once()
    assert storage._client is None
    storage.close()  # idempotent, no raise


def test_reset_propagates_non_notfound_error(storage: CosmosDBNoSqlStorage) -> None:
    storage._database.delete_container.side_effect = RuntimeError("throttled")
    with pytest.raises(RuntimeError, match="throttled"):
        storage.reset()


def test_reset_ignores_container_not_found(storage: CosmosDBNoSqlStorage) -> None:
    import azure.cosmos as ac

    storage._database.delete_container.side_effect = (
        ac.exceptions.CosmosResourceNotFoundError()
    )
    storage.reset()  # must not raise; recreates the container
    storage._database.create_container_if_not_exists.assert_called()


def test_get_scope_info_uses_server_side_aggregates(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    mock_container.query_items.side_effect = [
        iter([{"n": 2, "oldest": "2024-01-01T00:00:00", "newest": "2024-02-01T00:00:00"}]),
        iter(["k1", "k2"]),
        iter(["/a", "/a/b"]),
    ]
    info = storage.get_scope_info("/a")
    assert info.record_count == 2
    assert info.categories == ["k1", "k2"]
    agg_sql = mock_container.query_items.call_args_list[0].kwargs["query"]
    assert "COUNT(1)" in agg_sql and "MIN(c.created_at)" in agg_sql


def test_list_categories_uses_group_by(
    storage: CosmosDBNoSqlStorage, mock_container: MagicMock
) -> None:
    mock_container.query_items.return_value = iter(
        [{"category": "k1", "n": 3}, {"category": "k2", "n": 1}]
    )
    counts = storage.list_categories()
    assert counts == {"k1": 3, "k2": 1}
    sql = mock_container.query_items.call_args.kwargs["query"]
    assert "GROUP BY cat" in sql

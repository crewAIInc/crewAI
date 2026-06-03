"""Tests for Oracle embedding provider and callable."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.oracle.embedding_callable import (
    OracleEmbeddingFunction,
)
from crewai.rag.embeddings.providers.oracle.oracle_provider import OracleProvider


class FakeCursor:
    """Minimal Oracle cursor test double."""

    def __init__(
        self,
        rows: list[tuple[str]] | None = None,
        execute_side_effects: list[Exception | None] | None = None,
    ):
        self.rows = rows or []
        self.execute_side_effects = list(execute_side_effects or [])
        self.executed: list[tuple[str, Any, dict[str, Any]]] = []
        self.closed = False
        self.inputsizes: tuple[Any, ...] | None = None

    def execute(self, sql: str, params: Any = None, **kwargs: Any) -> None:
        self.executed.append((sql, params, kwargs))
        if self.execute_side_effects:
            side_effect = self.execute_side_effects.pop(0)
            if side_effect is not None:
                raise side_effect

    def setinputsizes(self, *args: Any) -> None:
        self.inputsizes = args

    def close(self) -> None:
        self.closed = True

    def __iter__(self):
        return iter(self.rows)


class FakeVectorArrayType:
    """Minimal Oracle named type test double."""

    def __init__(self):
        self.newobject_calls: list[list[str]] = []

    def newobject(self, values: list[str]) -> list[str]:
        self.newobject_calls.append(values)
        return values


class FakeConnection:
    """Minimal Oracle connection test double."""

    def __init__(self, cursor: FakeCursor | None = None):
        self._cursor = cursor or FakeCursor()
        self.vector_array_type = FakeVectorArrayType()
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def gettype(self, name: str) -> FakeVectorArrayType:
        assert name == "SYS.VECTOR_ARRAY_T"
        return self.vector_array_type

    def close(self) -> None:
        self.closed = True


class FakeOracleModule:
    """Minimal oracledb module test double."""

    DB_TYPE_JSON = object()

    def __init__(self, connection: FakeConnection | None = None):
        self.defaults = SimpleNamespace(fetch_lobs=True)
        self._connection = connection or FakeConnection()
        self.connect_calls: list[dict[str, Any]] = []

    def connect(self, **kwargs: Any) -> FakeConnection:
        self.connect_calls.append(kwargs)
        return self._connection


def _embedding_row(vector: list[float]) -> tuple[str]:
    return (json.dumps({"embed_vector": json.dumps(vector)}),)


class TestOracleProvider:
    """Unit tests for the Oracle provider."""

    @patch("crewai.rag.embeddings.factory.import_and_validate_definition")
    def test_build_embedder_oracle_routes_to_provider(self, mock_import: MagicMock):
        mock_provider_class = MagicMock()
        mock_provider_instance = MagicMock()
        mock_embedding_function = MagicMock()

        mock_import.return_value = mock_provider_class
        mock_provider_class.return_value = mock_provider_instance
        mock_provider_instance.embedding_callable.return_value = mock_embedding_function

        config = {
            "provider": "oracle",
            "config": {
                "connection_params": {"user": "u", "password": "p", "dsn": "db"},
                "embedding_params": {"provider": "database", "model": "demo_model"},
            },
        }

        result = build_embedder(config)

        assert result == mock_embedding_function
        mock_import.assert_called_once_with(
            "crewai.rag.embeddings.providers.oracle.oracle_provider.OracleProvider"
        )
        call_kwargs = mock_provider_class.call_args.kwargs
        assert call_kwargs["connection_params"] == {
            "user": "u",
            "password": "p",
            "dsn": "db",
        }
        assert call_kwargs["embedding_params"] == {
            "provider": "database",
            "model": "demo_model",
        }

    def test_provider_requires_exactly_one_connection_source(self):
        with pytest.raises(ValueError, match="exactly one of 'conn' or 'connection_params'"):
            OracleProvider(embedding_params={"provider": "database", "model": "m"})

        with pytest.raises(ValueError, match="exactly one of 'conn' or 'connection_params'"):
            OracleProvider(
                conn=object(),
                connection_params={"user": "u", "password": "p", "dsn": "db"},
                embedding_params={"provider": "database", "model": "m"},
            )

    def test_build_embedder_with_existing_connection_reuses_it(self, monkeypatch: pytest.MonkeyPatch):
        fake_conn = FakeConnection(
            cursor=FakeCursor(rows=[_embedding_row([0.1, 0.2]), _embedding_row([0.3, 0.4])])
        )
        fake_oracledb = FakeOracleModule()
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = build_embedder(
            {
                "provider": "oracle",
                "config": {
                    "conn": fake_conn,
                    "embedding_params": {"provider": "database", "model": "demo_model"},
                    "proxy": "http://proxy.example:8080",
                },
            }
        )

        embeddings = embedder(["hello", "world"])

        assert fake_oracledb.connect_calls == []
        assert len(embeddings) == 2
        assert embeddings[0].tolist() == pytest.approx([0.1, 0.2])
        assert embeddings[1].tolist() == pytest.approx([0.3, 0.4])
        assert fake_oracledb.defaults.fetch_lobs is True
        assert fake_conn.vector_array_type.newobject_calls
        proxy_call, embed_call, cleanup_call = fake_conn.cursor().executed
        assert "utl_http.set_proxy" in proxy_call[0]
        assert "dbms_vector_chain.utl_to_embeddings" in embed_call[0]
        assert "utl_http.set_proxy(NULL)" in cleanup_call[0]
        assert fake_conn.cursor().inputsizes == (None, fake_oracledb.DB_TYPE_JSON)
        assert fake_conn.cursor().closed is True

    def test_oracle_embedding_restores_fetch_lobs_when_call_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake_cursor = FakeCursor(execute_side_effects=[RuntimeError("embedding failed")])
        fake_conn = FakeConnection(cursor=fake_cursor)
        fake_oracledb = FakeOracleModule()
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = OracleEmbeddingFunction(
            conn=fake_conn,
            embedding_params={"provider": "database", "model": "demo_model"},
        )

        with pytest.raises(RuntimeError, match="embedding failed"):
            embedder(["hello"])

        assert fake_oracledb.defaults.fetch_lobs is True
        assert fake_cursor.closed is True

    def test_oracle_embedding_proxy_cleanup_error_preserves_original_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake_cursor = FakeCursor(
            execute_side_effects=[
                None,
                RuntimeError("embedding failed"),
                RuntimeError("cleanup failed"),
            ]
        )
        fake_conn = FakeConnection(cursor=fake_cursor)
        fake_oracledb = FakeOracleModule()
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = OracleEmbeddingFunction(
            conn=fake_conn,
            embedding_params={"provider": "database", "model": "demo_model"},
            proxy="http://proxy.example:8080",
        )

        with pytest.raises(RuntimeError, match="embedding failed"):
            embedder(["hello"])

        assert "utl_http.set_proxy(NULL)" in fake_cursor.executed[-1][0]
        assert fake_cursor.closed is True

    def test_build_embedder_connects_with_connection_params(self, monkeypatch: pytest.MonkeyPatch):
        fake_cursor = FakeCursor(rows=[_embedding_row([1.0, 2.0, 3.0])])
        fake_conn = FakeConnection(cursor=fake_cursor)
        fake_oracledb = FakeOracleModule(connection=fake_conn)
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = build_embedder(
            {
                "provider": "oracle",
                "config": {
                    "connection_params": {
                        "user": "oracle_user",
                        "password": "oracle_pass",
                        "dsn": "dbhost/service",
                        "config_dir": "/wallet",
                    },
                    "embedding_params": {"provider": "database", "model": "demo_model"},
                },
            }
        )

        embeddings = embedder(["database"])

        assert fake_oracledb.connect_calls == [
            {
                "user": "oracle_user",
                "password": "oracle_pass",
                "dsn": "dbhost/service",
                "config_dir": "/wallet",
            }
        ]
        assert len(embeddings) == 1
        assert embeddings[0].tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_build_embedder_without_oracledb_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delitem(sys.modules, "oracledb", raising=False)

        real_import = __import__

        def _import(name: str, *args: Any, **kwargs: Any):
            if name == "oracledb":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            with pytest.raises(ImportError, match="oracledb is required for oracle embeddings"):
                build_embedder(
                    {
                        "provider": "oracle",
                        "config": {
                            "connection_params": {"user": "u", "password": "p", "dsn": "db"},
                            "embedding_params": {"provider": "database", "model": "demo_model"},
                        },
                    }
                )

    def test_embedding_callable_name(self):
        assert OracleEmbeddingFunction.name() == "oracle"

    def test_build_embedder_with_string_input_and_none_row_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake_cursor = FakeCursor(rows=[None, _embedding_row([0.5, 0.6])])
        fake_conn = FakeConnection(cursor=fake_cursor)
        fake_oracledb = FakeOracleModule()
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = build_embedder(
            {
                "provider": "oracle",
                "config": {
                    "conn": fake_conn,
                    "embedding_params": {"provider": "database", "model": "demo_model"},
                },
            }
        )

        with pytest.raises(ValueError, match="empty row"):
            embedder("hello")

        assert len(fake_conn.vector_array_type.newobject_calls[0]) == 1
        assert fake_conn.vector_array_type.newobject_calls[0][0] == json.dumps(
            {"chunk_id": 1, "chunk_data": "hello"}
        )

    def test_build_embedder_with_empty_input_raises_without_db_call(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        fake_cursor = FakeCursor()
        fake_conn = FakeConnection(cursor=fake_cursor)
        fake_oracledb = FakeOracleModule()
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = build_embedder(
            {
                "provider": "oracle",
                "config": {
                    "conn": fake_conn,
                    "embedding_params": {"provider": "database", "model": "demo_model"},
                },
            }
        )

        with pytest.raises(ValueError, match="input cannot be empty"):
            embedder([])

        assert fake_cursor.executed == []

    def test_owned_connection_is_closed_on_delete(self, monkeypatch: pytest.MonkeyPatch):
        fake_conn = FakeConnection(cursor=FakeCursor(rows=[_embedding_row([1.0])]))
        fake_oracledb = FakeOracleModule(connection=fake_conn)
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = build_embedder(
            {
                "provider": "oracle",
                "config": {
                    "connection_params": {"user": "u", "password": "p", "dsn": "db"},
                    "embedding_params": {"provider": "database", "model": "demo_model"},
                },
            }
        )
        assert fake_conn.closed is False

        embedder.__del__()

        assert fake_conn.closed is True

    def test_delete_swallows_connection_close_error(self, monkeypatch: pytest.MonkeyPatch):
        class BrokenConnection(FakeConnection):
            def close(self) -> None:
                raise RuntimeError("boom")

        fake_oracledb = FakeOracleModule(connection=BrokenConnection())
        monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)

        embedder = build_embedder(
            {
                "provider": "oracle",
                "config": {
                    "connection_params": {"user": "u", "password": "p", "dsn": "db"},
                    "embedding_params": {"provider": "database", "model": "demo_model"},
                },
            }
        )

        embedder.__del__()

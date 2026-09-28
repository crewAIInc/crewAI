"""Schema reflection regressions for NL2SQLTool (issue #7782).

SQLite tests use real databases. PostgreSQL unit tests exercise real dialect
types with mocked reflection; live tests require CREWAI_TEST_PG_URI and a user
allowed to create tables, views, and types in public.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


pytest.importorskip("sqlalchemy")

from crewai_tools.tools.nl2sql.nl2sql_tool import NL2SQLTool
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql.base import PGInspector
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.types import NullType, TypeEngine


_MODULE = "crewai_tools.tools.nl2sql.nl2sql_tool"


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Iterator[tuple[str, Engine]]:
    uri = f"sqlite:///{tmp_path / 'example.db'}"
    engine = create_engine(uri)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)"))
            conn.execute(text("CREATE TABLE orders (id INTEGER, user_id INTEGER)"))
            conn.execute(text("INSERT INTO users VALUES (1, 'john'), (2, 'jane')"))
        yield uri, engine
    finally:
        engine.dispose()


def test_sqlite_construction_and_schema(sqlite_db: tuple[str, Engine]) -> None:
    uri, _ = sqlite_db
    tool = NL2SQLTool(db_uri=uri)

    assert sorted(tool.tables, key=lambda row: row["table_name"]) == [
        {"table_name": "orders"},
        {"table_name": "users"},
    ]
    assert tool.columns == {
        "users_columns": [
            {"column_name": "id", "data_type": "INTEGER"},
            {"column_name": "name", "data_type": "TEXT"},
        ],
        "orders_columns": [
            {"column_name": "id", "data_type": "INTEGER"},
            {"column_name": "user_id", "data_type": "INTEGER"},
        ],
    }
    assert tool.db_uri == uri
    assert tool.allow_dml is False
    assert set(tool.args_schema.model_fields) == {"sql_query"}


def test_sqlite_select_and_read_only_validation(sqlite_db: tuple[str, Engine]) -> None:
    uri, engine = sqlite_db
    tool = NL2SQLTool(db_uri=uri)
    # Own the query engine so the existing execute_sql pooling behavior cannot
    # leave a Windows file lock in this test. Introspection is unpatched.
    with patch(f"{_MODULE}.create_engine", return_value=engine):
        assert tool._run("SELECT id, name FROM users ORDER BY id") == [
            {"id": 1, "name": "john"},
            {"id": 2, "name": "jane"},
        ]
        assert tool.execute_sql("SELECT name FROM users WHERE id = :id", {"id": 2}) == [
            {"name": "jane"}
        ]
        with pytest.raises(ValueError, match="read-only mode"):
            tool._run("DELETE FROM users")


@pytest.mark.parametrize("uri", ["sqlite://", "sqlite:///:memory:"])
def test_empty_in_memory_sqlite(uri: str) -> None:
    tool = NL2SQLTool(db_uri=uri)
    assert tool.tables == []
    assert tool.columns == {}
    assert tool._run("SELECT 1 AS value") == [{"value": 1}]


def test_empty_sqlite_file(tmp_path: Path) -> None:
    tool = NL2SQLTool(db_uri=f"sqlite:///{tmp_path / 'empty.db'}")
    assert tool.tables == []
    assert tool.columns == {}


def test_sqlite_shared_memory_survives_introspection() -> None:
    uri = f"sqlite:///file:{uuid4().hex}?mode=memory&cache=shared&uri=true"
    engine = create_engine(uri)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE TABLE users (id INTEGER)"))
            conn.execute(text("INSERT INTO users VALUES (1)"))
            conn.commit()
            tool = NL2SQLTool(db_uri=uri)
            assert tool.tables == [{"table_name": "users"}]
            assert tool.columns == {
                "users_columns": [{"column_name": "id", "data_type": "INTEGER"}]
            }
            assert conn.execute(text("SELECT id FROM users")).scalar_one() == 1
    finally:
        engine.dispose()


def test_sqlite_untyped_columns(sqlite_db: tuple[str, Engine]) -> None:
    uri, engine = sqlite_db
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE untyped (id INTEGER, payload)"))
    tool = NL2SQLTool(db_uri=uri)
    assert tool.columns["untyped_columns"] == [
        {"column_name": "id", "data_type": "INTEGER"},
        {"column_name": "payload", "data_type": "UNKNOWN"},
    ]


def test_sqlite_views(sqlite_db: tuple[str, Engine]) -> None:
    uri, engine = sqlite_db
    with engine.begin() as conn:
        conn.execute(
            text("CREATE VIEW user_names AS SELECT name, 1 AS constant FROM users")
        )
    tool = NL2SQLTool(db_uri=uri)
    assert {"table_name": "user_names"} in tool.tables
    assert tool.columns["user_names_columns"] == [
        {"column_name": "name", "data_type": "TEXT"},
        {"column_name": "constant", "data_type": "UNKNOWN"},
    ]


@pytest.mark.parametrize(
    "table_name",
    ["CamelCase", "order details", "select", 'a"b', "users'; DROP TABLE users; --"],
)
def test_sqlite_identifiers_are_reflected_safely(
    sqlite_db: tuple[str, Engine], table_name: str
) -> None:
    uri, engine = sqlite_db
    Table(
        table_name,
        MetaData(),
        Column("id", Integer),
        Column('display "name"', String(25)),
    ).create(engine)
    tool = NL2SQLTool(db_uri=uri)
    assert {"table_name": table_name} in tool.tables
    assert tool.columns[f"{table_name}_columns"] == [
        {"column_name": "id", "data_type": "INTEGER"},
        {"column_name": 'display "name"', "data_type": "VARCHAR(25)"},
    ]
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one() == 2


@pytest.fixture
def postgres_reflection() -> Iterator[tuple[NL2SQLTool, MagicMock, MagicMock]]:
    tool = NL2SQLTool(db_uri="sqlite://")
    tool.db_uri = "postgresql://unused/test"
    engine = MagicMock(spec=Engine)
    engine.dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    inspector = MagicMock(spec=PGInspector)
    inspector.get_table_names.return_value = []
    inspector.get_view_names.return_value = []
    inspector.get_foreign_table_names.return_value = []
    with (
        patch(f"{_MODULE}.create_engine", return_value=engine),
        patch(f"{_MODULE}.inspect", return_value=inspector),
    ):
        yield tool, engine, inspector


def test_postgres_preserves_public_tables_views_and_foreign_tables(
    postgres_reflection: tuple[NL2SQLTool, MagicMock, MagicMock],
) -> None:
    tool, engine, inspector = postgres_reflection
    inspector.default_schema_name = "custom_schema"
    inspector.get_table_names.return_value = ["accounts"]
    inspector.get_view_names.return_value = ["account_summary"]
    inspector.get_foreign_table_names.return_value = ["external_accounts"]
    assert tool._fetch_available_tables() == [
        {"table_name": "accounts"},
        {"table_name": "account_summary"},
        {"table_name": "external_accounts"},
    ]
    inspector.get_table_names.assert_called_once_with(schema="public")
    inspector.get_view_names.assert_called_once_with(schema="public")
    inspector.get_foreign_table_names.assert_called_once_with(schema="public")
    engine.dispose.assert_called_once_with()


@pytest.mark.parametrize(
    ("column_type", "expected"),
    [
        (postgresql.INTEGER(), "INTEGER"),
        (postgresql.BIGINT(), "BIGINT"),
        (postgresql.VARCHAR(255), "VARCHAR(255)"),
        (postgresql.NUMERIC(10, 2), "NUMERIC(10, 2)"),
        (postgresql.TIMESTAMP(), "TIMESTAMP WITHOUT TIME ZONE"),
        (postgresql.TIMESTAMP(timezone=True), "TIMESTAMP WITH TIME ZONE"),
        (
            postgresql.TIMESTAMP(timezone=True, precision=3),
            "TIMESTAMP(3) WITH TIME ZONE",
        ),
        (postgresql.JSON(), "JSON"),
        (postgresql.JSONB(), "JSONB"),
        (postgresql.ARRAY(Integer()), "INTEGER[]"),
        (postgresql.ARRAY(postgresql.VARCHAR(25)), "VARCHAR(25)[]"),
        (postgresql.ENUM("new", "done", name="state"), "state"),
        (postgresql.ENUM("new", name="State", schema="custom"), 'custom."State"'),
        (postgresql.ARRAY(postgresql.ENUM("new", name="state")), "state[]"),
        (
            postgresql.DOMAIN("positive_int", Integer(), schema="custom"),
            "custom.positive_int",
        ),
        (postgresql.UUID(), "UUID"),
        (postgresql.BYTEA(), "BYTEA"),
        (postgresql.DOUBLE_PRECISION(), "DOUBLE PRECISION"),
        (postgresql.INET(), "INET"),
        (postgresql.INT4RANGE(), "INT4RANGE"),
        (postgresql.INTERVAL(precision=2), "INTERVAL (2)"),
        (NullType(), "UNKNOWN"),
        (postgresql.ARRAY(NullType()), "UNKNOWN"),
    ],
)
def test_postgres_type_rendering(
    postgres_reflection: tuple[NL2SQLTool, MagicMock, MagicMock],
    column_type: TypeEngine[Any],
    expected: str,
) -> None:
    tool, engine, inspector = postgres_reflection
    inspector.get_columns.return_value = [
        {"name": "value", "type": column_type},
        {"name": "id", "type": Integer()},
    ]
    assert tool._fetch_all_available_columns("accounts") == [
        {"column_name": "value", "data_type": expected},
        {"column_name": "id", "data_type": "INTEGER"},
    ]
    inspector.get_columns.assert_called_once_with("accounts", schema="public")
    engine.dispose.assert_called_once_with()


def test_introspection_does_not_use_query_execution(
    sqlite_db: tuple[str, Engine],
) -> None:
    uri, _ = sqlite_db
    with patch.object(
        NL2SQLTool, "execute_sql", side_effect=AssertionError("unexpected query")
    ):
        tool = NL2SQLTool(db_uri=uri)
    assert len(tool.tables) == 2
    assert len(tool.columns["users_columns"]) == 2


@pytest.mark.parametrize(
    "helper", ["_fetch_available_tables", "_fetch_all_available_columns"]
)
def test_invalid_uri_returns_error(helper: str) -> None:
    tool = NL2SQLTool(db_uri="sqlite://")
    tool.db_uri = "not-a-valid-scheme://x"
    result = getattr(tool, helper)(*(["users"] if helper.endswith("columns") else []))
    assert isinstance(result, str)
    assert result.startswith("Failed to create engine:")


@pytest.mark.parametrize(
    "failure",
    [
        "inspect",
        "get_table_names",
        "get_view_names",
        "get_foreign_table_names",
        "get_columns",
    ],
)
def test_reflection_failure_disposes_owned_engine(
    postgres_reflection: tuple[NL2SQLTool, MagicMock, MagicMock], failure: str
) -> None:
    tool, engine, inspector = postgres_reflection
    if failure != "inspect":
        getattr(inspector, failure).side_effect = SQLAlchemyError("reflection failure")
    with patch(f"{_MODULE}.inspect", return_value=inspector) as inspect_mock:
        if failure == "inspect":
            inspect_mock.side_effect = SQLAlchemyError("inspection failure")
        if failure == "get_columns":
            result = tool._fetch_all_available_columns("accounts")
            assert isinstance(result, str)
            assert result.startswith("Failed to fetch columns for accounts:")
        else:
            with pytest.raises(RuntimeError, match="Failed to fetch tables:"):
                tool.model_post_init(None)
    engine.dispose.assert_called_once_with()


def test_reflection_closes_connections_and_leaves_setup_engine_usable(
    sqlite_db: tuple[str, Engine],
) -> None:
    uri, setup_engine = sqlite_db
    created: list[Engine] = []
    disposed: list[Engine] = []
    opened: list[Any] = []
    closed: list[Any] = []
    setup_disposed = MagicMock()
    event.listen(setup_engine, "engine_disposed", setup_disposed)

    def tracking_create_engine(db_uri: str) -> Engine:
        engine = create_engine(db_uri)
        event.listen(engine, "engine_disposed", disposed.append)
        event.listen(engine, "connect", lambda conn, record: opened.append(conn))
        event.listen(engine, "close", lambda conn, record: closed.append(conn))
        created.append(engine)
        return engine

    try:
        with patch(f"{_MODULE}.create_engine", side_effect=tracking_create_engine):
            NL2SQLTool(db_uri=uri)
        assert created
        assert disposed == created
        assert opened and opened == closed
        setup_disposed.assert_not_called()
        with setup_engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one() == 2
    finally:
        for engine in created:
            engine.dispose()


@pytest.fixture
def pg_objects() -> Iterator[tuple[str, Engine, Table, str, str]]:
    uri = os.environ.get("CREWAI_TEST_PG_URI")
    if not uri:
        pytest.skip("set CREWAI_TEST_PG_URI to run live PostgreSQL tests")
    engine = create_engine(uri)
    suffix = uuid4().hex
    table_name, view_name, enum_name = (
        f"nl2sql_probe_{suffix}",
        f"nl2sql_view_{suffix}",
        f"nl2sql_state_{suffix}",
    )
    metadata = MetaData(schema="public")
    table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String(100)),
        Column("price", postgresql.NUMERIC(10, 2)),
        Column("created_at", postgresql.TIMESTAMP(timezone=True)),
        Column("updated_at", postgresql.TIMESTAMP()),
        Column("document", postgresql.JSON()),
        Column("payload", postgresql.JSONB()),
        Column("tags", postgresql.ARRAY(String(25))),
        Column(
            "state", postgresql.ENUM("new", "done", name=enum_name, schema="public")
        ),
        Column("identifier", postgresql.UUID()),
    )
    created = False
    try:
        with engine.begin() as conn:
            metadata.create_all(conn)
            conn.execute(table.insert().values(id=1, name="a"))
            # Names are generated UUID identifiers, never values from the URI.
            conn.execute(
                text(
                    f"CREATE VIEW public.{view_name} AS SELECT id, name FROM public.{table_name}"  # noqa: S608
                )
            )
        created = True
        yield uri, engine, table, view_name, enum_name
    finally:
        try:
            if created:
                with engine.begin() as conn:
                    conn.execute(text(f"DROP VIEW public.{view_name}"))
                    metadata.drop_all(conn)
        finally:
            engine.dispose()


def test_live_postgres_schema_types_and_select(
    pg_objects: tuple[str, Engine, Table, str, str],
) -> None:
    uri, engine, table, view_name, enum_name = pg_objects
    tool = NL2SQLTool(db_uri=uri)
    names = {entry["table_name"] for entry in tool.tables}
    assert {table.name, view_name} <= names
    columns = tool.columns[f"{table.name}_columns"]
    assert isinstance(columns, list)
    rendered = {entry["column_name"]: entry["data_type"] for entry in columns}
    enum_rendering = rendered.pop("state")
    assert enum_rendering in {enum_name, f"public.{enum_name}"}
    assert rendered == {
        "id": "INTEGER",
        "name": "VARCHAR(100)",
        "price": "NUMERIC(10, 2)",
        "created_at": "TIMESTAMP WITH TIME ZONE",
        "updated_at": "TIMESTAMP WITHOUT TIME ZONE",
        "document": "JSON",
        "payload": "JSONB",
        "tags": "VARCHAR(25)[]",
        "identifier": "UUID",
    }
    assert tool.columns[f"{view_name}_columns"] == [
        {"column_name": "id", "data_type": "INTEGER"},
        {"column_name": "name", "data_type": "VARCHAR(100)"},
    ]
    with patch(f"{_MODULE}.create_engine", return_value=engine):
        # The fixture generates this identifier from a UUID.
        query = f"SELECT id, name FROM public.{view_name}"  # noqa: S608
        assert tool._run(query) == [{"id": 1, "name": "a"}]


def test_live_postgres_public_scope_with_different_search_path(
    pg_objects: tuple[str, Engine, Table, str, str],
) -> None:
    uri, _, table, view_name, _ = pg_objects
    url = make_url(uri)
    if url.get_driver_name() not in {"psycopg2", "psycopg"}:
        pytest.skip("search_path connection options require a psycopg driver")
    options = url.query.get("options", "")
    assert isinstance(options, str)
    url = url.update_query_dict(
        {"options": f"{options} -csearch_path=pg_catalog".strip()}
    )
    tool = NL2SQLTool(db_uri=url.render_as_string(hide_password=False))
    names = {entry["table_name"] for entry in tool.tables}
    assert {table.name, view_name} <= names
    assert "pg_class" not in names
    columns = tool.columns[f"{table.name}_columns"]
    assert isinstance(columns, list)
    assert {entry["column_name"] for entry in columns} == set(table.columns.keys())

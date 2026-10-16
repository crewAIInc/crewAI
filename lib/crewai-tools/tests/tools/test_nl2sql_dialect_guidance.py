"""Tests for dialect-specific NL2SQLTool generation guidance."""

from unittest.mock import patch

import pytest


pytest.importorskip("sqlalchemy")

from crewai_tools.tools.nl2sql.nl2sql_tool import NL2SQLTool  # noqa: E402


def _make_tool(db_uri: str, **kwargs: object) -> NL2SQLTool:
    with (
        patch.object(NL2SQLTool, "_fetch_available_tables", return_value=[]),
        patch.object(NL2SQLTool, "_fetch_all_available_columns", return_value=[]),
    ):
        return NL2SQLTool(db_uri=db_uri, **kwargs)


def test_sqlite_dialect_is_inferred_from_uri() -> None:
    tool = _make_tool("sqlite:///example.db")

    assert tool.dialect == "sqlite"
    assert "Generate SQLite-compatible SQL" in tool.description
    assert "Do not use ILIKE" in tool.description


def test_postgresql_dialect_is_inferred_from_uri() -> None:
    tool = _make_tool("postgresql://user:password@localhost/database")

    assert tool.dialect == "postgresql"
    assert "Generate PostgreSQL-compatible SQL" in tool.description
    assert "ILIKE" in tool.description


def test_explicit_dialect_overrides_uri() -> None:
    tool = _make_tool(
        "postgresql://user:password@localhost/database", dialect=" SQLite "
    )

    assert tool.dialect == "sqlite"
    assert "Generate SQLite-compatible SQL" in tool.description


def test_custom_description_is_preserved() -> None:
    tool = _make_tool(
        "sqlite:///example.db",
        description="Use the reporting database.",
    )

    assert tool.description.startswith("Use the reporting database.")
    assert "Generate SQLite-compatible SQL" in tool.description


def test_dialect_guidance_is_not_duplicated_on_restore() -> None:
    tool = _make_tool("sqlite:///example.db")

    restored = _make_tool(**tool.model_dump())

    assert restored.description.count("Generate SQLite-compatible SQL") == 1

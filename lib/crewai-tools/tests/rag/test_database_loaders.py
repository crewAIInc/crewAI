from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.rag.base_loader import BaseLoader
from crewai_tools.rag.loaders.mysql_loader import MySQLLoader
from crewai_tools.rag.loaders.postgres_loader import PostgresLoader
from crewai_tools.rag.source_content import SourceContent


def connect_kwargs(loader: BaseLoader, module: str, db_uri: str) -> Mapping[str, Any]:
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [{"id": 1}]
    with patch(
        f"crewai_tools.rag.loaders.{module}.connect", return_value=connection
    ) as connect:
        loader.load(SourceContent("SELECT * FROM users"), metadata={"db_uri": db_uri})
    return connect.call_args.kwargs


LOADERS = [
    pytest.param(MySQLLoader, "mysql_loader", "mysql", id="mysql"),
    pytest.param(PostgresLoader, "postgres_loader", "postgresql", id="postgres"),
]


@pytest.mark.parametrize(("loader_class", "module", "scheme"), LOADERS)
def test_percent_encoded_credentials_are_decoded(
    loader_class: type[BaseLoader], module: str, scheme: str
) -> None:
    kwargs = connect_kwargs(
        loader_class(), module, f"{scheme}://app%40corp:p%40ss%2Fw%3Ard+1@db.local/shop"
    )

    assert kwargs["user"] == "app@corp"
    assert kwargs["password"] == "p@ss/w:rd+1"
    assert kwargs["host"] == "db.local"
    assert kwargs["database"] == "shop"


@pytest.mark.parametrize(("loader_class", "module", "scheme"), LOADERS)
def test_missing_credentials_stay_none(
    loader_class: type[BaseLoader], module: str, scheme: str
) -> None:
    kwargs = connect_kwargs(loader_class(), module, f"{scheme}://db.local/shop")

    assert kwargs["user"] is None
    assert kwargs["password"] is None


@pytest.mark.parametrize(("loader_class", "module", "scheme"), LOADERS)
def test_user_without_password(
    loader_class: type[BaseLoader], module: str, scheme: str
) -> None:
    kwargs = connect_kwargs(loader_class(), module, f"{scheme}://app@db.local/shop")

    assert kwargs["user"] == "app"
    assert kwargs["password"] is None

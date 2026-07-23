from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.mysql_search_tool.mysql_search_tool import MySQLSearchTool
from crewai_tools.tools.rag.rag_tool import RagTool


@pytest.fixture
def mock_rag_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.get_or_create_collection = MagicMock(return_value=None)
    mock_client.add_documents = MagicMock(return_value=None)
    mock_client.search = MagicMock(return_value=[])
    return mock_client


def create_mysql_search_tool(
    mock_rag_client: MagicMock, table_name: str
) -> MySQLSearchTool:
    with (
        patch(
            "crewai_tools.adapters.crewai_rag_adapter.get_rag_client",
            return_value=mock_rag_client,
        ),
        patch(
            "crewai_tools.adapters.crewai_rag_adapter.create_client",
            return_value=mock_rag_client,
        ),
    ):
        return MySQLSearchTool(
            db_uri="mysql://user:password@localhost:3306/test_database",
            table_name=table_name,
        )


@pytest.mark.parametrize(
    ("table_name", "expected_query"),
    [
        ("users", "SELECT * FROM `users`;"),
        ("user_profiles_2026", "SELECT * FROM `user_profiles_2026`;"),
        ("schema_name.users", "SELECT * FROM `schema_name`.`users`;"),
        ("information_schema.tables", "SELECT * FROM `information_schema`.`tables`;"),
    ],
)
def test_mysql_search_tool_quotes_valid_table_identifiers(
    mock_rag_client: MagicMock, table_name: str, expected_query: str
) -> None:
    with patch.object(RagTool, "add", return_value=None) as mock_add:
        create_mysql_search_tool(mock_rag_client, table_name)

    mock_add.assert_called_once_with(
        expected_query,
        data_type=DataType.MYSQL,
        metadata={"db_uri": "mysql://user:password@localhost:3306/test_database"},
    )


@pytest.mark.parametrize(
    "table_name",
    [
        "users where 1=1",
        "users; drop table users;--",
        "users -- comment",
        "users/*comment*/",
        "`users`",
        "schema.users.extra",
        "schema.",
        ".users",
        "123users",
    ],
)
def test_mysql_search_tool_rejects_invalid_table_identifiers(
    mock_rag_client: MagicMock, table_name: str
) -> None:
    with (
        patch.object(RagTool, "add", return_value=None) as mock_add,
        pytest.raises(ValueError, match="MySQL table_name must be a valid"),
    ):
        create_mysql_search_tool(mock_rag_client, table_name)

    mock_add.assert_not_called()


def test_mysql_search_tool_still_runs_search_queries(
    mock_rag_client: MagicMock,
) -> None:
    with patch.object(RagTool, "add", return_value=None):
        tool = create_mysql_search_tool(mock_rag_client, "users")

    with patch.object(RagTool, "_run", return_value="Alice") as mock_run:
        result = tool._run("alice")

    assert "Alice" in result
    mock_run.assert_called_once_with(
        query="alice", similarity_threshold=None, limit=None
    )


def test_mysql_search_tool_uses_mysql_data_type_metadata(
    mock_rag_client: MagicMock,
) -> None:
    with patch.object(RagTool, "add", return_value=None) as mock_add:
        create_mysql_search_tool(mock_rag_client, "users")

    assert mock_add.call_args.kwargs == {
        "data_type": DataType.MYSQL,
        "metadata": {"db_uri": "mysql://user:password@localhost:3306/test_database"},
    }

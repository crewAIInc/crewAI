import pytest

pytest.importorskip("singlestoredb")
pytest.importorskip("sqlalchemy")

from crewai_tools.tools.singlestore_search_tool import (  # noqa: E402
    singlestore_search_tool as module,
)


@pytest.fixture
def no_table_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid connecting to a database while inspecting connection args."""
    monkeypatch.setattr(module, "SINGLSTORE_AVAILABLE", True)
    monkeypatch.setattr(
        module.SingleStoreSearchTool, "_initialize_tables", lambda self, tables: None
    )


def test_parse_json_false_is_preserved(no_table_setup: None) -> None:
    """An explicit ``parse_json=False`` must reach the driver.

    ``parse_json or True`` evaluated to ``True`` for every value, so callers
    could never turn JSON parsing off.
    """
    tool = module.SingleStoreSearchTool(
        host="localhost",
        user="user",
        password="password",
        database="db",
        parse_json=False,
    )

    assert tool.connection_args["parse_json"] is False


def test_parse_json_defaults_to_true(no_table_setup: None) -> None:
    tool = module.SingleStoreSearchTool(
        host="localhost",
        user="user",
        password="password",
        database="db",
    )

    assert tool.connection_args["parse_json"] is True

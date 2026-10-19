import importlib
import json
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

from crewai_tools import PixeltableSearchTool
from pydantic import ValidationError
import pytest


def _table(rows: list[dict[str, Any]], columns: list[str]) -> MagicMock:
    table = MagicMock()
    similarity = MagicMock(name="similarity")
    for name in columns:
        column = MagicMock(name=name)
        column.similarity.return_value = similarity
        setattr(table, name, column)
    ordered = MagicMock()
    limited = MagicMock()
    selected = MagicMock()
    table.order_by.return_value = ordered
    ordered.limit.return_value = limited
    limited.select.return_value = selected
    selected.collect.return_value = rows
    return table


def test_search_returns_rows_with_scores() -> None:
    rows = [
        {"text": "alpha docs", "score": 0.91},
        {"text": "beta docs", "score": 0.2},
    ]
    table = _table(rows, ["text"])
    tool = PixeltableSearchTool(table=table, column="text")

    result = json.loads(tool.run(query="alpha"))

    assert result == rows
    table.text.similarity.assert_called_once_with(string="alpha")
    table.order_by.assert_called_once_with(
        table.text.similarity.return_value, asc=False
    )
    table.order_by.return_value.limit.assert_called_once_with(5)
    table.order_by.return_value.limit.return_value.select.assert_called_once_with(
        table.text, score=table.text.similarity.return_value
    )


def test_search_passes_index_limit_and_extra_columns() -> None:
    rows = [{"text": "alpha", "title": "A", "score": Decimal("0.5")}]
    table = _table(rows, ["text", "title"])
    tool = PixeltableSearchTool(
        table=table,
        column="text",
        limit=2,
        select=["title", "text"],
        idx="minilm",
    )

    result = json.loads(tool.run(query="alpha"))

    assert result == [{"text": "alpha", "title": "A", "score": 0.5}]
    table.text.similarity.assert_called_once_with(string="alpha", idx="minilm")
    table.order_by.return_value.limit.assert_called_once_with(2)
    table.order_by.return_value.limit.return_value.select.assert_called_once_with(
        table.text,
        table.title,
        score=table.text.similarity.return_value,
    )


def test_table_path_is_loaded_with_get_table(monkeypatch: pytest.MonkeyPatch) -> None:
    table = _table([{"text": "row", "score": 1.0}], ["text"])
    pixeltable = MagicMock()
    pixeltable.get_table.return_value = table
    monkeypatch.setattr(
        "crewai_tools.tools.pixeltable_search_tool.pixeltable_search_tool._import_pixeltable",
        lambda: pixeltable,
    )
    tool = PixeltableSearchTool(table=" dir.docs ", column=" text ")

    assert json.loads(tool.run(query="hello")) == [{"text": "row", "score": 1.0}]
    pixeltable.get_table.assert_called_once_with("dir.docs")
    assert tool.column == "text"


def test_missing_pixeltable_explains_how_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def _import(name: str, package: str | None = None) -> Any:
        if name == "pixeltable":
            raise ImportError("not installed")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _import)
    tool = PixeltableSearchTool(table="dir.docs", column="text")

    with pytest.raises(ImportError, match=r"crewai-tools\[pixeltable\]"):
        tool.run(query="hello")


def test_missing_column_raises_value_error() -> None:
    tool = PixeltableSearchTool(table=object(), column="text")

    with pytest.raises(ValueError, match="text"):
        tool.run(query="hello")


def test_non_dict_rows_raise_type_error() -> None:
    table = _table([], ["text"])
    table.order_by.return_value.limit.return_value.select.return_value.collect.return_value = [
        "not-a-row"
    ]
    tool = PixeltableSearchTool(table=table, column="text")

    with pytest.raises(TypeError, match="row dictionaries"):
        tool.run(query="hello")


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValidationError):
        PixeltableSearchTool(table="   ", column="text")
    with pytest.raises(ValidationError):
        PixeltableSearchTool(table="dir.docs", column="  ")
    with pytest.raises(ValidationError):
        PixeltableSearchTool(table="dir.docs", column="text", limit=0)
    with pytest.raises(ValidationError):
        PixeltableSearchTool(table="dir.docs", column="text", select=["title", " "])
    with pytest.raises(ValidationError):
        PixeltableSearchTool(table="dir.docs", column="text", idx=" ")


def test_package_dependency_and_query_schema() -> None:
    tool = PixeltableSearchTool(table="dir.docs", column="text")
    schema = tool.args_schema.model_json_schema()

    assert tool.package_dependencies == ["pixeltable"]
    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]

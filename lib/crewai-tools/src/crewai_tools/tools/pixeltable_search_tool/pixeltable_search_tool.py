from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import importlib
import json
from types import ModuleType
from typing import Any, cast

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, field_validator


_INSTALL_HINT = (
    "The 'pixeltable' package is required to use PixeltableSearchTool. "
    "Install it with `uv add 'crewai-tools[pixeltable]'` or "
    "`pip install 'crewai-tools[pixeltable]'`. "
    "Pixeltable requires Python 3.11 or newer."
)


class PixeltableSearchToolSchema(BaseModel):
    """Input for a Pixeltable similarity search."""

    query: str = Field(
        ...,
        description="Text to compare against the indexed Pixeltable column.",
    )


class PixeltableSearchTool(BaseTool):
    """Similarity search over a Pixeltable table.

    The configured text column must already have an embedding index. Pixeltable
    embeds the query with that index and returns the closest rows.
    """

    name: str = "PixeltableSearchTool"
    description: str = (
        "Search a Pixeltable table for rows whose text is similar to a query. "
        "The configured text column must have an embedding index. "
        "Returns a JSON list of matching rows and similarity scores."
    )
    args_schema: type[BaseModel] = PixeltableSearchToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["pixeltable"])

    table: str | Any = Field(
        ...,
        description=(
            "Pixeltable table to search. Either a table path such as 'dir.docs' "
            "or a pixeltable.Table instance."
        ),
    )
    column: str = Field(
        ...,
        description="Name of the text column that has an embedding index.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        description="Maximum number of rows to return.",
    )
    select: list[str] | None = Field(
        default=None,
        description="Extra column names to return along with the text column and score.",
    )
    idx: str | None = Field(
        default=None,
        description=(
            "Embedding index name. Required only when the column has more than "
            "one embedding index."
        ),
    )

    @field_validator("table", mode="before")
    @classmethod
    def _validate_table(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("table path must be a non-empty string.")
            return stripped
        if value is None:
            raise ValueError("table must be a Pixeltable path or Table.")
        return value

    @field_validator("column")
    @classmethod
    def _validate_column(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("column must be a non-empty column name.")
        return stripped

    @field_validator("select")
    @classmethod
    def _validate_select(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for name in value:
            stripped = name.strip()
            if not stripped:
                raise ValueError("select column names must be non-empty.")
            cleaned.append(stripped)
        return cleaned

    @field_validator("idx")
    @classmethod
    def _validate_idx(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("idx must be a non-empty embedding index name.")
        return stripped

    def _run(self, query: str) -> str:
        """Return the closest rows as a JSON list.

        Args:
            query: Text to compare against the indexed column.

        Returns:
            A JSON list of row objects. Each object includes the indexed text
            column, any extra selected columns, and a ``score``.
        """
        table = self._load_table()
        column_names = list(dict.fromkeys([self.column, *(self.select or [])]))
        columns = [_column(table, name) for name in column_names]
        searched_column = columns[0]

        similarity_kwargs: dict[str, Any] = {"string": query}
        if self.idx is not None:
            similarity_kwargs["idx"] = self.idx
        similarity = searched_column.similarity(**similarity_kwargs)

        result = (
            table.order_by(similarity, asc=False)
            .limit(self.limit)
            .select(*columns, score=similarity)
            .collect()
        )
        return _format_results(result)

    def _load_table(self) -> Any:
        if not isinstance(self.table, str):
            return self.table

        return _import_pixeltable().get_table(self.table)


def _import_pixeltable() -> ModuleType:
    try:
        return importlib.import_module("pixeltable")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _column(table: Any, name: str) -> Any:
    try:
        return getattr(table, name)
    except AttributeError as exc:
        raise ValueError(
            f"Column {name!r} was not found on the Pixeltable table."
        ) from exc


def _format_results(result: Any) -> str:
    rows = list(result)
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(
                "Pixeltable search results must be row dictionaries. "
                f"Got {type(row).__name__}."
            )
    return json.dumps(rows, indent=2, default=_json_default)


def _json_default(value: object) -> object:
    """Convert Pixeltable cell values that ``json`` cannot encode."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = cast(object, tolist())
        if type(converted) is not type(value):
            return converted

    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = cast(object, item())
        except (TypeError, ValueError):
            return str(value)
        if type(converted) is not type(value):
            return converted

    return str(value)

"""Tests to validate markdown table formatting in documentation files.

These tests ensure that markdown tables in the documentation are well-formed
and render correctly. Specifically, they check that:
- All rows have the same number of columns as the header
- Pipe characters inside table cells are properly escaped
- Separator rows match the header column count
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"

DOCS_TABLE_FILES = [
    "en/concepts/tasks.mdx",
    "pt-BR/concepts/tasks.mdx",
    "ko/concepts/tasks.mdx",
]


def _split_table_row(line: str) -> list[str]:
    """Split a markdown table row on unescaped pipe characters.

    Escaped pipes (``\\|``) are preserved as literal ``|`` inside cells.
    """
    cells = re.split(r"(?<!\\)\|", line)
    return [cell.replace("\\|", "|").strip() for cell in cells]


def _parse_markdown_tables(content: str) -> list[tuple[int, list[list[str]]]]:
    """Parse all markdown tables from content.

    Returns a list of (start_line_number, table_rows) tuples.
    Each table_rows is a list of rows, where each row is a list of cell values.
    """
    lines = content.split("\n")
    tables: list[tuple[int, list[list[str]]]] = []
    current_table: list[list[str]] = []
    table_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not current_table:
                table_start = i + 1
            cells = _split_table_row(stripped)
            cells = cells[1:-1]
            current_table.append(cells)
        else:
            if current_table:
                tables.append((table_start, current_table))
                current_table = []

    if current_table:
        tables.append((table_start, current_table))

    return tables


def _is_separator_row(cells: list[str]) -> bool:
    """Check if a row is a table separator (e.g., | :--- | :--- |)."""
    return all(re.match(r"^:?-+:?$", cell.strip()) for cell in cells if cell.strip())


@pytest.mark.parametrize("doc_path", DOCS_TABLE_FILES)
def test_markdown_tables_have_consistent_columns(doc_path: str) -> None:
    """Verify all rows in each markdown table have the same number of columns."""
    full_path = DOCS_DIR / doc_path
    if not full_path.exists():
        pytest.skip(f"Doc file not found: {full_path}")

    content = full_path.read_text(encoding="utf-8")
    tables = _parse_markdown_tables(content)

    for table_start, rows in tables:
        if len(rows) < 2:
            continue
        header_col_count = len(rows[0])
        for row_idx, row in enumerate(rows[1:], start=1):
            assert len(row) == header_col_count, (
                f"Table at line {table_start} in {doc_path}: "
                f"row {row_idx + 1} has {len(row)} columns, expected {header_col_count}. "
                f"Row content: {'|'.join(row)}"
            )


@pytest.mark.parametrize("doc_path", DOCS_TABLE_FILES)
def test_task_attributes_table_has_no_unescaped_pipes_in_cells(doc_path: str) -> None:
    """Verify the Task Attributes table doesn't have unescaped pipe chars in cells.

    The '|' character is the column delimiter in markdown tables. If a type
    annotation like `List[Callable] | List[str]` contains an unescaped pipe,
    it will be interpreted as a column separator and break the table layout.
    """
    full_path = DOCS_DIR / doc_path
    if not full_path.exists():
        pytest.skip(f"Doc file not found: {full_path}")

    content = full_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    in_task_attrs = False
    for i, line in enumerate(lines):
        if "## Task Attributes" in line or "## Atributos da Tarefa" in line:
            in_task_attrs = True
            continue

        if in_task_attrs and line.startswith("##"):
            break

        if in_task_attrs and line.strip().startswith("|") and line.strip().endswith("|"):
            stripped = line.strip()
            cells = stripped.split("|")
            cells = cells[1:-1]

            if _is_separator_row(cells):
                continue

            for cell_idx, cell in enumerate(cells):
                unescaped_pipes = re.findall(r"(?<!\\)\|", cell)
                assert not unescaped_pipes, (
                    f"Line {i + 1} in {doc_path}, cell {cell_idx + 1}: "
                    f"found unescaped pipe character in cell content: '{cell.strip()}'. "
                    f"Use '\\|' to escape pipe characters inside table cells."
                )


@pytest.mark.parametrize("doc_path", DOCS_TABLE_FILES)
def test_task_attributes_table_separator_matches_header(doc_path: str) -> None:
    """Verify the separator row has the same number of columns as the header."""
    full_path = DOCS_DIR / doc_path
    if not full_path.exists():
        pytest.skip(f"Doc file not found: {full_path}")

    content = full_path.read_text(encoding="utf-8")
    tables = _parse_markdown_tables(content)

    for table_start, rows in tables:
        if len(rows) < 2:
            continue

        header = rows[0]
        separator = rows[1]

        if _is_separator_row(separator):
            assert len(separator) == len(header), (
                f"Table at line {table_start} in {doc_path}: "
                f"separator has {len(separator)} columns but header has {len(header)}. "
                f"This usually means the header or separator row has extra '|' delimiters."
            )

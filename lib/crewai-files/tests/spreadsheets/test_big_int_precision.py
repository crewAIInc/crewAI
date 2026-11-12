"""Integers wider than float64's ~15 significant digits must survive reading,
filtering and rendering exactly (#9380)."""

from crewai_files.spreadsheets.answers import Filter, Query, query
from crewai_files.spreadsheets.reading import parse


CSV = (
    b"order_id,amount\n"
    b"1234567890123456789,10\n"
    b"1234567890123456790,20\n"
    b"1234567890123456791,5\n"
)


def test_rows_render_distinct_ids_exactly():
    workbook = parse(CSV, "orders.csv")
    out = query(workbook, "orders.csv", Query(operation="rows"))
    for value in ("1234567890123456789", "1234567890123456790", "1234567890123456791"):
        assert value in out
    assert "1234567890123456768" not in out  # the float64-rounded artifact


def test_eq_filter_matches_exactly_one_row():
    workbook = parse(CSV, "orders.csv")
    out = query(
        workbook,
        "orders.csv",
        Query(
            operation="rows",
            where=[Filter(column="order_id", op="eq", values=["1234567890123456790"])],
        ),
    )
    assert "1 of 3 rows match" in out


def test_value_counts_sees_three_distinct():
    workbook = parse(CSV, "orders.csv")
    out = query(
        workbook,
        "orders.csv",
        Query(operation="value_counts", columns=["order_id"]),
    )
    assert "3 distinct" in out


def test_short_integers_still_render_as_numbers():
    workbook = parse(b"n\n42\n7\n", "nums.csv")
    out = query(workbook, "nums.csv", Query(operation="rows"))
    assert "42" in out and "7" in out


def test_mixed_native_and_digit_column_keeps_both():
    """Excel stores native floats; a mixed ID column (native cell + digit text)
    must keep both when the exact-int path rebuilds the series (Cursor Bugbot
    HIGH on #9389)."""
    csv = b"order_id,amount\n1234567890123456789,10\n42,20\n1234567890123456791,5\n"
    workbook = parse(csv, "mixed.csv")
    out = query(workbook, "mixed.csv", Query(operation="rows"))
    assert "42" in out
    assert "1234567890123456789" in out
    assert "1234567890123456791" in out


def test_huge_digit_column_does_not_break_queries():
    """A pure-digit column whose sum overflows float64 must not turn every
    query on the sheet into an OverflowError (strix LOW on #9389)."""
    csv = b"order_id\n" + b"\n".join(str(10**308 + i).encode() for i in range(3)) + b"\n"
    workbook = parse(csv, "huge.csv")
    out = query(workbook, "huge.csv", Query(operation="rows"))
    assert "10" in out  # the digits render


def test_empty_cells_render_blank_in_exact_int_columns():
    """pd.NA cells must render blank, not the literal <NA> (Cursor Bugbot
    MEDIUM on #9389)."""
    csv = b"order_id,amount\n1234567890123456789,\n42,20\n"
    workbook = parse(csv, "na.csv")
    out = query(workbook, "na.csv", Query(operation="rows"))
    assert "<NA>" not in out

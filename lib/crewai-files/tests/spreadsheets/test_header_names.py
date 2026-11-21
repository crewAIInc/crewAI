"""Header-name tests that do not require the optional xlsx reader."""

from crewai_files.spreadsheets.reading import parse


def test_generated_header_suffixes_do_not_collide_with_real_headers() -> None:
    """A literal "Amount (2)" header must not erase a duplicated column."""
    book = parse(b"Amount,Amount,Amount (2)\n1,2,3\n", "w.csv")

    assert list(book.sheets[0].table().frame.columns) == [
        "Amount",
        "Amount (2)",
        "Amount (2) (2)",
    ]

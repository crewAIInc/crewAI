import pytest

from crewai.utilities.string_utils import slugify


@pytest.mark.parametrize(
    ("text", "separator", "expected"),
    [
        ("sand", "and", "sand"),
        ("box x x", "x", "boxxxxx"),
        ("hello world", "\\", "hello\\world"),
        ("hello world", r"\1", r"hello\1world"),
        ("  Hello, world!  ", "_", "hello_world"),
        ("  Hello, world!  ", "--", "hello--world"),
        ("Café's menu", "-", "cafes-menu"),
        ("hello world", "", "helloworld"),
        ("!!!", "_", ""),
        ("", "_", ""),
    ],
)
def test_slugify_uses_literal_separator(
    text: str, separator: str, expected: str
) -> None:
    assert slugify(text, separator=separator) == expected

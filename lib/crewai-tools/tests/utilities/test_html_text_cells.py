import pytest
from bs4 import BeautifulSoup

from crewai_tools.utilities.html_text import visible_text


def _text(html: str) -> str:
    return visible_text(BeautifulSoup(html, "html.parser"))


@pytest.mark.parametrize(
    "html",
    [
        "<table><tr><td>a</td><td>b</td></tr></table>",
        "<table>\n  <tr>\n    <td>a</td>\n    <td>b</td>\n  </tr>\n</table>",
        "<table><tr><td><p>a</p></td><td><p>b</p></td></tr></table>",
        "<table>\n<tr>\n<td>\n<p>a</p>\n</td>\n<td>\n<p>b</p>\n</td>\n</tr>\n</table>",
        "<table><tr><td>a </td><td> b</td></tr></table>",
    ],
)
def test_table_row_stays_on_one_line_with_single_space(html):
    assert _text(html).strip() == "a b"


def test_table_rows_are_separate_lines():
    html = (
        "<table><tr><th>h1</th><th>h2</th></tr>"
        "<tr><td><p>a</p></td><td><p>b</p></td></tr></table>"
    )
    assert _text(html).strip() == "h1 h2\na b"


def test_br_inside_cell_still_breaks_line():
    assert _text("<table><tr><td>a<br>b</td><td>c</td></tr></table>").strip() == (
        "a\nb c"
    )
import time

from bs4 import BeautifulSoup
import pytest

from crewai_tools.utilities.html_text import visible_text


def text_of(html: str) -> str:
    return visible_text(BeautifulSoup(html, "html.parser"))


def test_inline_markup_does_not_split_words():
    assert text_of("<p>un<b>believ</b>able</p>") == "unbelievable"


def test_punctuation_after_a_link_stays_attached():
    assert text_of('<p>see <a href="/docs">the docs</a>.</p>') == "see the docs."


def test_whitespace_between_inline_elements_is_kept():
    assert text_of("<p><b>one</b> <i>two</i></p>") == "one two"


def test_blocks_start_new_lines_without_blank_lines():
    html = "<h1>Setup</h1><p>Body</p><ul><li>one</li><li>two</li></ul>"
    assert text_of(html) == "Setup\nBody\none\ntwo"


def test_br_starts_a_new_line():
    assert text_of("<p>one<br>two<br/>three</p>") == "one\ntwo\nthree"


def test_table_cells_are_separated_by_a_space_and_rows_by_a_line():
    html = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>"
    assert text_of(html) == "a b\nc d"


def test_pretty_printed_html_has_no_stray_whitespace():
    html = "<div>\n  <p>one </p>\n  <p> two</p>\n</div>\n"
    assert text_of(html) == "one\ntwo"


def test_title_does_not_run_into_the_body_text():
    html = "<html><head><title>Title</title></head><body>Hello <b>there</b></body></html>"
    assert text_of(html) == "Title\nHello there"


def test_comments_doctype_script_and_style_are_not_read():
    html = (
        "<!DOCTYPE html><html><head><style>p{color:red}</style></head>"
        "<body><!-- hidden --><p>Hi</p><script>x=1</script></body></html>"
    )
    assert text_of(html) == "Hi"


def test_entities_are_decoded():
    assert text_of("<p>a &amp; b &lt; c</p>") == "a & b < c"


@pytest.mark.parametrize("html", ["", "<p></p>", "<div> </div>", "<br>"])
def test_empty_documents_give_empty_text(html):
    assert text_of(html) == ""


def test_the_tree_is_not_modified():
    soup = BeautifulSoup("<p>a<br>b</p><td>c</td>", "html.parser")
    before = str(soup)
    visible_text(soup)
    assert str(soup) == before


def test_deeply_nested_markup_does_not_exhaust_the_stack():
    depth = 5000
    soup = BeautifulSoup(
        "<div>" * depth + "deep" + "</div>" * depth, "html.parser"
    )
    assert visible_text(soup) == "deep"


def test_many_sibling_blocks_are_extracted_in_linear_time():
    soup = BeautifulSoup("<p>x</p>" * 20000 + "<hr>" * 20000, "html.parser")
    start = time.perf_counter()
    result = visible_text(soup)
    elapsed = time.perf_counter() - start

    assert result.count("x") == 20000
    assert elapsed < 2

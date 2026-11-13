import pytest

from crewai_tools.rag.chunkers.base_chunker import RecursiveCharacterTextSplitter


@pytest.mark.parametrize(
    ("text", "chunk_size", "separators", "expected"),
    [
        ("abcdef", 3, [""], ["abc", "def"]),
        ("aa\nbb\ncc", 5, ["\n", ""], ["aa\nbb", "\ncc"]),
        ("ab--cd--ef", 6, ["--", ""], ["ab--cd", "--ef"]),
    ],
)
def test_zero_overlap_preserves_text_without_exceeding_size(
    text: str, chunk_size: int, separators: list[str], expected: list[str]
) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, separators=separators
    )
    chunks = splitter.split_text(text)
    assert chunks == expected
    assert "".join(chunks) == text
    assert all(len(chunk) <= chunk_size for chunk in chunks)


def test_overlap_does_not_make_next_chunk_exceed_size() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=2, separators=[" ", ""])
    chunks = splitter.split_text("abcd efgh ijkl")
    assert all(len(chunk) <= 5 for chunk in chunks)
    assert "abcd" in chunks[0]
    assert any("efgh" in chunk for chunk in chunks)
    assert any("ijkl" in chunk for chunk in chunks)


def test_character_chunks_keep_requested_overlap() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3, chunk_overlap=1, separators=[""])
    assert splitter.split_text("abcdefg") == ["abc", "cde", "efg"]


def test_removed_separators_count_toward_chunk_size() -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=5, chunk_overlap=0, separators=[" ", ""], keep_separator=False
    )
    assert splitter.split_text("aa bb cc") == ["aa bb", "cc"]

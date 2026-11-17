from crewai_tools.rag.chunkers.base_chunker import RecursiveCharacterTextSplitter


def test_keep_separator_does_not_duplicate_newlines():
    """The split step already attaches separators; merging must not add them again."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

    assert splitter.split_text("a\nb") == ["a\nb"]
    assert splitter.split_text("Para one.\n\nPara two.") == [
        "Para one.\n\nPara two."
    ]
    assert splitter.split_text('{"a": 1}\n{"b": 2}') == ['{"a": 1}\n{"b": 2}']


def test_keep_separator_false_still_rejoins_with_the_separator():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, keep_separator=False)

    assert splitter.split_text("a\nb") == ["a\nb"]

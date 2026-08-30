from crewai_tools.rag.chunkers.base_chunker import BaseChunker


class TextChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n\n",  # Multiple line breaks (sections)
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                "",
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)


class DocxChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 250,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n\n",  # Multiple line breaks (major sections)
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                "",
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)


class MdxChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 300,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n## ",  # H2 headers (major sections)
                "\n### ",  # H3 headers (subsections)
                "\n#### ",  # H4 headers (sub-subsections)
                "\n\n",
                "\n```",
                "\n",
                ". ",
                "! ",
                "? ",
                "; ",
                ", ",
                " ",
                "",
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)

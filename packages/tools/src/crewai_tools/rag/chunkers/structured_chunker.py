from crewai_tools.rag.chunkers.base_chunker import BaseChunker


class CsvChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\nRow ",  # Row boundaries (from CSVLoader format)
                "\n",  # Line breaks
                " | ",  # Column separators
                ", ",  # Comma separators
                " ",  # Word breaks
                "",  # Character level
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)


class JsonChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n",  # Object/array boundaries
                "\n",  # Line breaks
                "},",  # Object endings
                "],",  # Array endings
                ", ",  # Property separators
                ": ",  # Key-value separators
                " ",  # Word breaks
                "",  # Character level
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)


class XmlChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 250,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n",  # Element boundaries
                "\n",  # Line breaks
                ">",  # Tag endings
                ". ",  # Sentence endings (for text content)
                "! ",  # Exclamation endings
                "? ",  # Question endings
                ", ",  # Comma separators
                " ",  # Word breaks
                "",  # Character level
            ]
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)

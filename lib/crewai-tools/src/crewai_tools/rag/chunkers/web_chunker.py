from crewai_tools.rag.chunkers.base_chunker import BaseChunker


class WebsiteChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 2500,
        chunk_overlap: int = 250,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        if separators is None:
            separators = [
                "\n\n\n",
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

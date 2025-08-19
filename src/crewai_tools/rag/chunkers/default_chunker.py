from crewai_tools.rag.chunkers.base_chunker import BaseChunker
from typing import List, Optional

class DefaultChunker(BaseChunker):
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 20, separators: Optional[List[str]] = None, keep_separator: bool = True):
        super().__init__(chunk_size, chunk_overlap, separators, keep_separator)

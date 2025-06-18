from pathlib import Path
from typing import Dict, List

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    pdfplumber = None

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries PDF file content using embeddings."""

    def __init__(self, *args, **kwargs):
        """Initialize PDFKnowledgeSource and check for pdfplumber availability."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PDF knowledge sources. "
                "Please install it with: pip install 'crewai[knowledge]'"
            )
        super().__init__(*args, **kwargs)

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess PDF file content."""
        if not PDFPLUMBER_AVAILABLE:
            return {}
        
        content = {}

        for path in self.safe_file_paths:
            text = ""
            path = self.convert_to_path(path)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            content[path] = text
        return content

    def _import_pdfplumber(self):
        """Dynamically import pdfplumber."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber is required for PDF knowledge sources. "
                "Please install it with: pip install 'crewai[knowledge]'"
            )
        return pdfplumber

    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        self._import_pdfplumber()
        
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

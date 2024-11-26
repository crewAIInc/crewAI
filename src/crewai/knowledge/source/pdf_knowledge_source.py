from typing import List, Dict
from pathlib import Path

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries PDF file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess PDF file content."""
        super().load_content()  # Validate the file paths
        pdfplumber = self._import_pdfplumber()

        paths = [self.file_path] if isinstance(self.file_path, Path) else self.file_path
        content = {}

        for path in paths:
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            content[path] = text
        return content

    def _import_pdfplumber(self):
        """Dynamically import pdfplumber."""
        try:
            import pdfplumber

            return pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is not installed. Please install it with: pip install pdfplumber"
            )

    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self.save_documents(metadata=self.metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

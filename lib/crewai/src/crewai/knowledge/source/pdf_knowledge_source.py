from pathlib import Path
from types import ModuleType

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries PDF file content using embeddings."""

    def load_content(self) -> dict[Path, str]:
        """Load and preprocess PDF file content."""
        pdfplumber = self._import_pdfplumber()

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

    def _import_pdfplumber(self) -> ModuleType:
        """Dynamically import pdfplumber."""
        try:
            import pdfplumber

            return pdfplumber
        except ImportError as e:
            raise ImportError(
                "pdfplumber is not installed. Please install it with: pip install pdfplumber"
            ) from e

    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        for text in self.content.values():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self._save_documents()

    async def aadd(self) -> None:
        """Add PDF file content asynchronously."""
        for text in self.content.values():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        await self._asave_documents()

    def _chunk_text(self, text: str) -> list[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

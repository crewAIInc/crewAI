from typing import List

from crewai.knowledge.embedder.base_embedder import BaseEmbedder
from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries PDF file content using embeddings."""

    def load_content(self) -> str:
        """Load and preprocess PDF file content."""
        super().load_content()  # Validate the file path
        pdfplumber = self._import_pdfplumber()
        text = ""
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    def _import_pdfplumber(self):
        """Dynamically import pdfplumber."""
        try:
            import pdfplumber

            return pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is not installed. Please install it with: pip install pdfplumber"
            )

    def add(self, embedder: BaseEmbedder) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        new_chunks = self._chunk_text(self.content)
        self.chunks.extend(new_chunks)
        # Compute embeddings for the new chunks
        new_embeddings = embedder.embed_chunks(new_chunks)
        # Save the embeddings
        self.chunk_embeddings.extend(new_embeddings)
        self._save_documents(metadata=self.metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

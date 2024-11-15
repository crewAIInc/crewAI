import csv
from typing import List

from crewai.knowledge.embedder.base_embedder import BaseEmbedder
from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries CSV file content using embeddings."""

    def load_content(self) -> str:
        """Load and preprocess CSV file content."""
        super().load_content()  # Validate the file path
        with open(self.file_path, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            content = ""
            for row in reader:
                content += " ".join(row) + "\n"
        return content

    def add(self, embedder: BaseEmbedder) -> None:
        """
        Add CSV file content to the knowledge source, chunk it, compute embeddings,
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

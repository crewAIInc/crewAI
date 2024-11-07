from typing import List

from crewai.knowledge.embedder.base_embedder import BaseEmbedder
from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class ExcelKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries Excel file content using embeddings."""

    def load_content(self) -> str:
        """Load and preprocess Excel file content."""
        super().load_content()  # Validate the file path
        pd = self._import_dependencies()
        df = pd.read_excel(self.file_path)
        content = df.to_csv(index=False)
        return content

    def _import_dependencies(self):
        """Dynamically import dependencies."""
        try:
            import openpyxl  # noqa
            import pandas as pd

            return pd
        except ImportError as e:
            missing_package = str(e).split()[-1]
            raise ImportError(
                f"{missing_package} is not installed. Please install it with: pip install {missing_package}"
            )

    def add(self, embedder: BaseEmbedder) -> None:
        """
        Add Excel file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        new_chunks = self._chunk_text(self.content)
        self.chunks.extend(new_chunks)
        # Compute embeddings for the new chunks
        new_embeddings = embedder.embed_chunks(new_chunks)
        # Save the embeddings
        self.chunk_embeddings.extend(new_embeddings)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

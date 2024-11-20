from typing import Dict, List
from pathlib import Path
from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class ExcelKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries Excel file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess Excel file content."""
        super().load_content()  # Validate the file path
        pd = self._import_dependencies()

        if isinstance(self.file_path, list):
            file_path = self.file_path[0]
        else:
            file_path = self.file_path

        df = pd.read_excel(file_path)
        content = df.to_csv(index=False)
        return {file_path: content}

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

    def add(self) -> None:
        """
        Add Excel file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        # Convert dictionary values to a single string if content is a dictionary
        if isinstance(self.content, dict):
            content_str = "\n".join(str(value) for value in self.content.values())
        else:
            content_str = str(self.content)

        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self.save_documents(metadata=self.metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

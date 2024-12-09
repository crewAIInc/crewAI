from pathlib import Path
from typing import Dict, List

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class ExcelKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries Excel file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess Excel file content."""
        pd = self._import_dependencies()

        content_dict = {}
        for file_path in self.safe_file_paths:
            file_path = self.convert_to_path(file_path)
            df = pd.read_excel(file_path)
            content = df.to_csv(index=False)
            content_dict[file_path] = content
        return content_dict

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
        self._save_documents()

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

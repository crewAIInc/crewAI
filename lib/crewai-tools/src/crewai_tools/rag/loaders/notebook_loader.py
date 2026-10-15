import json
from typing import Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class NotebookLoader(BaseLoader):
    """Extract nbformat 4 cell sources from a UTF-8 file or inline JSON.

    Code is never executed. Outputs and attachments are omitted.
    """

    def load(self, content: SourceContent, **kwargs: Any) -> LoaderResult:
        source_ref = content.source_ref
        text = content.source
        if content.path_exists():
            with open(source_ref, encoding="utf-8") as file:
                text = file.read()

        notebook = json.loads(text)
        cells = notebook["cells"]
        metadata = self._extract_metadata(notebook)
        metadata.update(
            cell_count=len(cells), markdown_cells=0, code_cells=0, raw_cells=0
        )
        parts = []
        for number, cell in enumerate(cells, start=1):
            cell_type = cell["cell_type"]
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            parts.append(f"[cell {number}: {cell_type}]\n{source}")
            metadata[f"{cell_type}_cells"] += 1

        text = "\n\n".join(parts)
        return LoaderResult(
            content=text,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text),
        )

    @staticmethod
    def _extract_metadata(notebook: dict[str, Any]) -> dict[str, Any]:
        metadata = {"format": "ipynb", "nbformat": notebook["nbformat"]}
        if "nbformat_minor" in notebook:
            metadata["nbformat_minor"] = notebook["nbformat_minor"]
        notebook_metadata = notebook.get("metadata", {})
        for field, key in (("language_info", "language"), ("kernelspec", "kernel")):
            name = notebook_metadata.get(field, {}).get("name")
            if isinstance(name, str):
                metadata[key] = name
        return metadata

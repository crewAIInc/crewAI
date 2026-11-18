import json
from typing import Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class NotebookLoader(BaseLoader):
    """Extract nbformat 4 cell sources from a UTF-8 file or inline JSON.

    Code is never executed. Outputs and attachments are omitted.
    Invalid cells raise ValueError with their 1-based cell number.
    """

    def load(self, content: SourceContent, **kwargs: Any) -> LoaderResult:
        """Return ordered cell source text, scalar metadata, and source identity."""
        source_ref = content.source_ref
        text = content.source
        if content.path_exists():
            with open(source_ref, encoding="utf-8-sig") as file:
                text = file.read()

        notebook = json.loads(text)
        if not isinstance(notebook, dict):
            raise ValueError("Expected a notebook object")
        version = notebook.get("nbformat")
        if not isinstance(version, int) or version != 4:
            raise ValueError("Only nbformat 4 notebooks are supported")
        cells = notebook.get("cells")
        if not isinstance(cells, list):
            raise ValueError("Notebook cells must be a list")
        metadata = self._extract_metadata(notebook)
        metadata.update(
            cell_count=len(cells), markdown_cells=0, code_cells=0, raw_cells=0
        )
        parts = []
        for number, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                raise ValueError(f"Invalid cell {number}: expected an object")
            cell_type = cell.get("cell_type")
            if cell_type not in ("markdown", "code", "raw"):
                raise ValueError(f"Invalid cell {number}: unsupported cell_type")
            source = cell.get("source")
            if isinstance(source, list) and all(
                isinstance(part, str) for part in source
            ):
                source = "".join(source)
            if not isinstance(source, str):
                raise ValueError(
                    f"Invalid cell {number}: source must be a string or list of strings"
                )
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
        """Retain notebook versions and valid language and kernel names."""
        metadata = {"format": "ipynb", "nbformat": notebook["nbformat"]}
        minor = notebook.get("nbformat_minor")
        if isinstance(minor, int) and not isinstance(minor, bool) and minor >= 0:
            metadata["nbformat_minor"] = minor
        notebook_metadata = notebook.get("metadata", {})
        if not isinstance(notebook_metadata, dict):
            return metadata
        for field, key in (("language_info", "language"), ("kernelspec", "kernel")):
            value = notebook_metadata.get(field)
            if isinstance(value, dict) and isinstance(value.get("name"), str):
                metadata[key] = value["name"]
        return metadata

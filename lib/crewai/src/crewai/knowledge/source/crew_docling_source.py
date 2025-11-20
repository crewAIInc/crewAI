from __future__ import annotations

from collections.abc import Iterator
import importlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# --- third-party/optional imports (OK to keep in try/except) ---
try:
    from docling.datamodel.base_models import (  # type: ignore[import-not-found]
        InputFormat,
    )
    from docling_core.transforms.chunker.hierarchical_chunker import (  # type: ignore[import-not-found]
        HierarchicalChunker,
    )

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# Ensure the converter module is present too; otherwise the flag is misleading.
if DOCLING_AVAILABLE:
    import importlib.util as _ilu

    if (
        _ilu.find_spec("docling.document_converter") is None
        or _ilu.find_spec("docling.exceptions") is None
    ):
        DOCLING_AVAILABLE = False

# --- regular imports must stay together, before any non-import statements ---
from pydantic import Field, PrivateAttr

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger


# Safe default; will be overwritten at runtime if docling is present
DoclingConversionError: type[BaseException] | None = None


class CrewDoclingSource(BaseKnowledgeSource):
    """Default Source class for converting documents to markdown or json
    This will auto support PDF, DOCX, and TXT, XLSX, Images, and HTML files without any additional dependencies and follows the docling package as the source of truth.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "The docling package is required to use CrewDoclingSource. "
                "Please install it using: uv add docling"
            )
        super().__init__(*args, **kwargs)

    _logger: Logger = Logger(verbose=True)

    file_path: list[Path | str] | None = Field(default=None)
    file_paths: list[Path | str] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    safe_file_paths: list[Path | str] = Field(default_factory=list)
    content: list[Any] = Field(default_factory=list)
    _aligned_paths: list[Path | str] = PrivateAttr(default_factory=list)
    document_converter: Any = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.file_path:
            self._logger.log(
                "warning",
                "The 'file_path' attribute is deprecated and will be removed in a future version. Please use 'file_paths' instead.",
                color="yellow",
            )
            self.file_paths = self.file_path

        self.safe_file_paths = self.validate_content()

        # Import docling pieces dynamically to avoid mypy missing-import issues.
        try:
            docling_mod = importlib.import_module("docling.document_converter")
        except Exception as e:
            raise ImportError(
                "docling is partially installed: 'docling.document_converter' not found."
                "Please install/upgrade docling: `uv add docling` ."
            ) from e
        document_converter_cls = docling_mod.DocumentConverter

        # Resolve ConversionError dynamically (no static import)
        try:
            exc_mod = importlib.import_module("docling.exceptions")
            exc_cls = getattr(exc_mod, "ConversionError", None)
            if isinstance(exc_cls, type) and issubclass(exc_cls, BaseException):
                global DoclingConversionError
                DoclingConversionError = exc_cls
            else:
                self._logger.log(
                    "warning",
                    "docling.exceptions.ConversionError not found or invalid; using generic handling.",
                    color="yellow",
                )
                DoclingConversionError = None
        except Exception as err:
            # Log instead of bare `pass` to satisfy ruff S110
            self._logger.log(
                "warning",
                f"docling.exceptions not available ({err!s}); using generic handling.",
                color="yellow",
            )
            DoclingConversionError = None

        self.document_converter = document_converter_cls(
            allowed_formats=[
                InputFormat.MD,
                InputFormat.ASCIIDOC,
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.IMAGE,
                InputFormat.XLSX,
                InputFormat.PPTX,
                InputFormat.CSV,
            ]
        )
        self.content = self._load_content()

    def _load_content(self) -> list[Any]:
        try:
            return self._convert_source_to_docling_documents()
        except Exception as e:
            if DoclingConversionError is not None and isinstance(
                e, DoclingConversionError
            ):
                self._logger.log(
                    "error",
                    f"Error loading content: {e}. Supported formats: {self.document_converter.allowed_formats}",
                    "red",
                )
            else:
                self._logger.log("error", f"Error loading content: {e}")
            raise

    def add(self) -> None:
        """Convert each document to chunks, attach filepath metadata, and persist."""
        if not self.content:
            return

        for filepath, doc in zip(self._aligned_paths, self.content, strict=True):
            chunk_idx = 0
            for chunk in self._chunk_doc(doc):
                self.chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "filepath": str(filepath),
                            "chunk_index": chunk_idx,
                            "source_type": "docling",
                        },
                    }
                )
                chunk_idx += 1

        self._save_documents()

    def _convert_one(self, fp: Path | str) -> tuple[Any, Path | str] | None:
        """Convert a single file; on failure, log and return None."""
        try:
            result = self.document_converter.convert(fp)
            return result.document, fp
        except Exception as e:
            if DoclingConversionError is not None and isinstance(
                e, DoclingConversionError
            ):
                self._logger.log(
                    "warning",
                    f"Skipping {fp!s}: conversion failed with {e!s}",
                    color="yellow",
                )
            else:
                self._logger.log(
                    "warning",
                    f"Skipping {fp!s}: unexpected error during conversion {e!s}",
                    color="yellow",
                )
            return None

    def _convert_source_to_docling_documents(self) -> list[Any]:
        """
        Convert files one-by-one to preserve (filepath, document) alignment.

        Any file that fails conversion is skipped (with a warning). For all successful
        conversions, we maintain a parallel list of source paths so the add() step can
        attach correct per-chunk filepath metadata without relying on zip truncation.
        """
        aligned_docs: list[Any] = []
        aligned_paths: list[Path | str] = []

        for fp in self.safe_file_paths:
            item = self._convert_one(fp)
            if item is None:
                continue
            doc, aligned_fp = item
            aligned_docs.append(doc)
            aligned_paths.append(aligned_fp)

        self._aligned_paths = aligned_paths
        return aligned_docs

    def _chunk_doc(self, doc: Any) -> Iterator[str]:
        chunker = HierarchicalChunker()
        for chunk in chunker.chunk(doc):
            yield chunk.text

    def validate_content(self) -> list[Path | str]:
        processed_paths: list[Path | str] = []
        for path in self.file_paths:
            if isinstance(path, str):
                if path.startswith(("http://", "https://")):
                    try:
                        if self._validate_url(path):
                            processed_paths.append(path)
                        else:
                            raise ValueError(f"Invalid URL format: {path}")
                    except Exception as e:
                        raise ValueError(f"Invalid URL: {path}. Error: {e!s}") from e
                else:
                    local_path = Path(KNOWLEDGE_DIRECTORY + "/" + path)
                    if local_path.exists():
                        processed_paths.append(local_path)
                    else:
                        raise FileNotFoundError(f"File not found: {local_path}")
            else:
                processed_paths.append(path)
        return processed_paths

    def _validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all(
                [
                    result.scheme in ("http", "https"),
                    result.netloc,
                    len(result.netloc.split(".")) >= 2,
                ]
            )
        except Exception:
            return False

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast
from urllib.parse import urlparse

from pydantic import Field, model_validator
from typing_extensions import Self

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger


if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc.document import DoclingDocument


_DOCLING_IMPORT_ERROR = (
    "The docling package is required to use CrewDoclingSource. "
    "Please install it using: uv add docling"
)


class _DoclingModules(NamedTuple):
    """Lazily-imported docling symbols used by ``CrewDoclingSource``."""

    input_format: Any
    document_converter: Any
    conversion_error: type[BaseException]
    hierarchical_chunker: Any


@cache
def _import_docling() -> _DoclingModules:
    """Import docling submodules lazily and cache the result.

    Raises:
        ImportError: If the docling package is not installed.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter
        from docling.exceptions import ConversionError
        from docling_core.transforms.chunker.hierarchical_chunker import (
            HierarchicalChunker,
        )
    except ImportError as e:
        raise ImportError(_DOCLING_IMPORT_ERROR) from e
    return _DoclingModules(
        input_format=InputFormat,
        document_converter=DocumentConverter,
        conversion_error=ConversionError,
        hierarchical_chunker=HierarchicalChunker,
    )


def _build_default_document_converter() -> DocumentConverter:
    """Construct the default ``DocumentConverter`` with crewAI's allowed formats."""
    docling = _import_docling()
    input_format = docling.input_format
    return cast(
        "DocumentConverter",
        docling.document_converter(
            allowed_formats=[
                input_format.MD,
                input_format.ASCIIDOC,
                input_format.PDF,
                input_format.DOCX,
                input_format.HTML,
                input_format.IMAGE,
                input_format.XLSX,
                input_format.PPTX,
            ]
        ),
    )


class CrewDoclingSource(BaseKnowledgeSource):
    """Default Source class for converting documents to markdown or json.

    This will auto support PDF, DOCX, and TXT, XLSX, Images, and HTML files without
    any additional dependencies and follows the docling package as the source of truth.
    """

    @model_validator(mode="before")
    @classmethod
    def _ensure_docling_available(cls, data: Any) -> Any:
        _import_docling()
        return data

    _logger: Logger = Logger(verbose=True)

    source_type: Literal["docling"] = "docling"
    file_path: list[Path | str] | None = Field(default=None)
    file_paths: list[Path | str] = Field(default_factory=list)
    chunks: list[str] = Field(default_factory=list)
    safe_file_paths: list[Path | str] = Field(default_factory=list)
    content: list[Any] = Field(default_factory=list)
    document_converter: Any = Field(default_factory=_build_default_document_converter)

    @model_validator(mode="after")
    def _load_sources(self) -> Self:
        if self.file_path:
            self._logger.log(
                "warning",
                "The 'file_path' attribute is deprecated and will be removed in a future version. Please use 'file_paths' instead.",
                color="yellow",
            )
            self.file_paths = self.file_path
        self.safe_file_paths = self.validate_content()
        self.content = self._load_content()
        return self

    def _load_content(self) -> list[DoclingDocument]:
        conversion_error = _import_docling().conversion_error
        try:
            return self._convert_source_to_docling_documents()
        except conversion_error as e:
            self._logger.log(
                "error",
                f"Error loading content: {e}. Supported formats: {self.document_converter.allowed_formats}",
                "red",
            )
            raise e
        except Exception as e:
            self._logger.log("error", f"Error loading content: {e}")
            raise e

    def add(self) -> None:
        if self.content is None:
            return
        for doc in self.content:
            new_chunks_iterable = self._chunk_doc(doc)
            self.chunks.extend(list(new_chunks_iterable))
        self._save_documents()

    async def aadd(self) -> None:
        """Add docling content asynchronously."""
        if self.content is None:
            return
        for doc in self.content:
            new_chunks_iterable = self._chunk_doc(doc)
            self.chunks.extend(list(new_chunks_iterable))
        await self._asave_documents()

    def _convert_source_to_docling_documents(self) -> list[DoclingDocument]:
        conv_results_iter = self.document_converter.convert_all(self.safe_file_paths)
        return [result.document for result in conv_results_iter]

    def _chunk_doc(self, doc: DoclingDocument) -> Iterator[str]:
        chunker = _import_docling().hierarchical_chunker()
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

"""Adapter for CrewAI's native RAG system."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast
import uuid

from crewai.rag.config.types import RagConfigType
from crewai.rag.config.utils import get_rag_client
from crewai.rag.core.base_client import BaseClient
from crewai.rag.factory import create_client
from crewai.rag.types import BaseRecord, SearchResult
from pydantic import PrivateAttr
from pydantic.dataclasses import is_pydantic_dataclass
from typing_extensions import TypeIs, Unpack

from crewai_tools.rag.data_types import DataType
from crewai_tools.rag.misc import sanitize_metadata_for_chromadb
from crewai_tools.tools.rag.rag_tool import Adapter
from crewai_tools.tools.rag.types import AddDocumentParams, ContentItem


if TYPE_CHECKING:
    from crewai.rag.qdrant.config import QdrantConfig


def _is_qdrant_config(config: Any) -> TypeIs[QdrantConfig]:
    """Check if config is a QdrantConfig using safe duck typing.

    Args:
        config: RAG configuration to check.

    Returns:
        True if config is a QdrantConfig instance.
    """
    if not is_pydantic_dataclass(config):
        return False

    try:
        return cast(bool, config.provider == "qdrant")  # type: ignore[attr-defined]
    except (AttributeError, ImportError):
        return False


class CrewAIRagAdapter(Adapter):
    """Adapter that uses CrewAI's native RAG system.

    Supports custom vector database configuration through the config parameter.
    """

    collection_name: str = "default"
    summarize: bool = False
    similarity_threshold: float = 0.6
    limit: int = 5
    config: RagConfigType | None = None
    _client: BaseClient | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the CrewAI RAG client after model initialization."""
        if self.config is not None:
            self._client = create_client(self.config)
        else:
            self._client = get_rag_client()
        collection_params: dict[str, Any] = {"collection_name": self.collection_name}

        if self.config is not None and _is_qdrant_config(self.config):
            if self.config.vectors_config is not None:
                collection_params["vectors_config"] = self.config.vectors_config
        self._client.get_or_create_collection(**collection_params)

    def query(
        self,
        question: str,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        """Query the knowledge base with a question.

        Args:
            question: The question to ask
            similarity_threshold: Minimum similarity score for results (default: 0.6)
            limit: Maximum number of results to return (default: 5)

        Returns:
            Relevant content from the knowledge base
        """
        search_limit = limit if limit is not None else self.limit
        search_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        if self._client is None:
            raise ValueError("Client is not initialized")

        results: list[SearchResult] = self._client.search(
            collection_name=self.collection_name,
            query=question,
            limit=search_limit,
            score_threshold=search_threshold,
        )

        if not results:
            return "No relevant content found."

        contents: list[str] = []
        for result in results:
            content: str = result.get("content", "")
            if content:
                contents.append(content)

        return "\n\n".join(contents)

    def add(self, *args: ContentItem, **kwargs: Unpack[AddDocumentParams]) -> None:
        """Add content to the knowledge base.

        Args:
            *args: Content items to add (strings, paths, or document dicts)
            **kwargs: Additional parameters including:
                - data_type: DataType enum or string (e.g., "file", "pdf_file", "text")
                - path: Path to file or directory (alternative to positional arg)
                - file_path: Alias for path
                - metadata: Additional metadata to attach to documents
                - url: URL to fetch content from
                - website: Website URL to scrape
                - github_url: GitHub repository URL
                - youtube_url: YouTube video URL
                - directory_path: Path to directory

        Examples:
            rag_tool.add("path/to/document.pdf", data_type=DataType.PDF_FILE)

            rag_tool.add(path="path/to/document.pdf", data_type="file")
            rag_tool.add(file_path="path/to/document.pdf", data_type="pdf_file")

            rag_tool.add("path/to/document.pdf")  # auto-detects PDF
        """
        import os

        from crewai_tools.rag.base_loader import LoaderResult
        from crewai_tools.rag.data_types import DataType, DataTypes
        from crewai_tools.rag.source_content import SourceContent

        documents: list[BaseRecord] = []
        raw_data_type = kwargs.get("data_type")
        base_metadata: dict[str, Any] = kwargs.get("metadata", {})

        data_type: DataType | None = None
        if raw_data_type is not None:
            if isinstance(raw_data_type, DataType):
                if raw_data_type != DataType.FILE:
                    data_type = raw_data_type
            elif isinstance(raw_data_type, str):
                if raw_data_type != "file":
                    try:
                        data_type = DataType(raw_data_type)
                    except ValueError:
                        raise ValueError(
                            f"Invalid data_type: '{raw_data_type}'. "
                            f"Valid values are: 'file' (auto-detect), or one of: "
                            f"{', '.join(dt.value for dt in DataType)}"
                        ) from None

        content_items: list[ContentItem] = list(args)

        path_value = kwargs.get("path") or kwargs.get("file_path")
        if path_value is not None:
            content_items.append(path_value)

        if url := kwargs.get("url"):
            content_items.append(url)
        if website := kwargs.get("website"):
            content_items.append(website)
        if github_url := kwargs.get("github_url"):
            content_items.append(github_url)
        if youtube_url := kwargs.get("youtube_url"):
            content_items.append(youtube_url)
        if directory_path := kwargs.get("directory_path"):
            content_items.append(directory_path)

        file_extensions = {
            ".pdf",
            ".txt",
            ".csv",
            ".json",
            ".xml",
            ".docx",
            ".mdx",
            ".md",
        }

        for arg in content_items:
            source_ref: str
            if isinstance(arg, dict):
                source_ref = str(arg.get("source", arg.get("content", "")))
            else:
                source_ref = str(arg)

            if not data_type:
                ext = os.path.splitext(source_ref)[1].lower()
                is_url = source_ref.startswith(("http://", "https://", "file://"))
                if (
                    ext in file_extensions
                    and not is_url
                    and not os.path.isfile(source_ref)
                ):
                    raise FileNotFoundError(f"File does not exist: {source_ref}")
                data_type = DataTypes.from_content(source_ref)

            if data_type == DataType.DIRECTORY:
                if not os.path.isdir(source_ref):
                    raise ValueError(f"Directory does not exist: {source_ref}")

                # Define binary and non-text file extensions to skip
                binary_extensions = {
                    ".pyc",
                    ".pyo",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".bmp",
                    ".ico",
                    ".svg",
                    ".webp",
                    ".pdf",
                    ".zip",
                    ".tar",
                    ".gz",
                    ".bz2",
                    ".7z",
                    ".rar",
                    ".exe",
                    ".dll",
                    ".so",
                    ".dylib",
                    ".bin",
                    ".dat",
                    ".db",
                    ".sqlite",
                    ".class",
                    ".jar",
                    ".war",
                    ".ear",
                }

                for root, dirs, files in os.walk(source_ref):
                    dirs[:] = [d for d in dirs if not d.startswith(".")]

                    for filename in files:
                        if filename.startswith("."):
                            continue

                        # Skip binary files based on extension
                        file_ext = os.path.splitext(filename)[1].lower()
                        if file_ext in binary_extensions:
                            continue

                        # Skip __pycache__ directories
                        if "__pycache__" in root:
                            continue

                        file_path: str = os.path.join(root, filename)
                        try:
                            file_data_type: DataType = DataTypes.from_content(file_path)
                            file_loader = file_data_type.get_loader()
                            file_chunker = file_data_type.get_chunker()

                            file_source = SourceContent(file_path)
                            file_result: LoaderResult = file_loader.load(file_source)

                            file_chunks = file_chunker.chunk(file_result.content)

                            for chunk_idx, file_chunk in enumerate(file_chunks):
                                file_metadata: dict[str, Any] = base_metadata.copy()
                                file_metadata.update(file_result.metadata)
                                file_metadata["data_type"] = str(file_data_type)
                                file_metadata["file_path"] = file_path
                                file_metadata["chunk_index"] = chunk_idx
                                file_metadata["total_chunks"] = len(file_chunks)

                                if isinstance(arg, dict):
                                    file_metadata.update(arg.get("metadata", {}))

                                chunk_hash = hashlib.sha256(
                                    f"{file_result.doc_id}_{chunk_idx}_{file_chunk}".encode()
                                ).hexdigest()
                                chunk_id = str(uuid.UUID(chunk_hash[:32]))

                                documents.append(
                                    {
                                        "doc_id": chunk_id,
                                        "content": file_chunk,
                                        "metadata": sanitize_metadata_for_chromadb(
                                            file_metadata
                                        ),
                                    }
                                )
                        except Exception:  # noqa: S112
                            # Silently skip files that can't be processed
                            continue
            else:
                metadata: dict[str, Any] = base_metadata.copy()
                source_content = SourceContent(source_ref)

                if data_type in [
                    DataType.PDF_FILE,
                    DataType.TEXT_FILE,
                    DataType.DOCX,
                    DataType.CSV,
                    DataType.JSON,
                    DataType.XML,
                    DataType.MDX,
                ]:
                    if not source_content.is_url() and not source_content.path_exists():
                        raise FileNotFoundError(f"File does not exist: {source_ref}")

                loader = data_type.get_loader()
                chunker = data_type.get_chunker()

                loader_result: LoaderResult = loader.load(source_content)

                chunks = chunker.chunk(loader_result.content)

                for i, chunk in enumerate(chunks):
                    chunk_metadata: dict[str, Any] = metadata.copy()
                    chunk_metadata.update(loader_result.metadata)
                    chunk_metadata["data_type"] = str(data_type)
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    chunk_metadata["source"] = source_ref

                    if isinstance(arg, dict):
                        chunk_metadata.update(arg.get("metadata", {}))

                    chunk_hash = hashlib.sha256(
                        f"{loader_result.doc_id}_{i}_{chunk}".encode()
                    ).hexdigest()
                    chunk_id = str(uuid.UUID(chunk_hash[:32]))

                    documents.append(
                        {
                            "doc_id": chunk_id,
                            "content": chunk,
                            "metadata": sanitize_metadata_for_chromadb(chunk_metadata),
                        }
                    )

        if documents:
            if self._client is None:
                raise ValueError("Client is not initialized")
            self._client.add_documents(
                collection_name=self.collection_name, documents=documents
            )

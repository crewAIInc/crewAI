"""Type definitions for RAG tool configuration."""

from pathlib import Path
from typing import Any, Literal, TypeAlias

from crewai.rag.embeddings.types import ProviderSpec
from typing_extensions import TypedDict

from crewai_tools.rag.data_types import DataType


DataTypeStr: TypeAlias = Literal[
    "file",
    "pdf_file",
    "text_file",
    "csv",
    "json",
    "xml",
    "docx",
    "mdx",
    "mysql",
    "postgres",
    "github",
    "directory",
    "website",
    "docs_site",
    "youtube_video",
    "youtube_channel",
    "text",
]

ContentItem: TypeAlias = str | Path | dict[str, Any]


class AddDocumentParams(TypedDict, total=False):
    """Parameters for adding documents to the RAG system."""

    data_type: DataType | DataTypeStr
    metadata: dict[str, Any]
    path: str | Path
    file_path: str | Path
    website: str
    url: str
    github_url: str
    youtube_url: str
    directory_path: str | Path


class VectorDbConfig(TypedDict):
    """Configuration for vector database provider.

    Attributes:
        provider: RAG provider literal.
        config: RAG configuration options.
    """

    provider: Literal["chromadb", "qdrant"]
    config: dict[str, Any]


class RagToolConfig(TypedDict, total=False):
    """Configuration accepted by RAG tools.

    Supports embedding model and vector database configuration.

    Attributes:
        embedding_model: Embedding model configuration accepted by RAG tools.
        vectordb: Vector database configuration accepted by RAG tools.
    """

    embedding_model: ProviderSpec
    vectordb: VectorDbConfig

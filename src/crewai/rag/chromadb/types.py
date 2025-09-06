"""Type definitions specific to ChromaDB implementation."""

from collections.abc import Mapping
from typing import Any, NamedTuple

from chromadb.api import AsyncClientAPI, ClientAPI
from chromadb.api.configuration import CollectionConfigurationInterface
from chromadb.api.types import (
    CollectionMetadata,
    DataLoader,
    Include,
    Loadable,
    Where,
    WhereDocument,
)
from chromadb.api.types import (
    EmbeddingFunction as ChromaEmbeddingFunction,
)
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from crewai.rag.core.base_client import BaseCollectionParams, BaseCollectionSearchParams

ChromaDBClientType = ClientAPI | AsyncClientAPI


class ChromaEmbeddingFunctionWrapper(ChromaEmbeddingFunction):
    """Base class for ChromaDB EmbeddingFunction to work with Pydantic validation."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for ChromaDB EmbeddingFunction.

        This allows Pydantic to handle ChromaDB's EmbeddingFunction type
        without requiring arbitrary_types_allowed=True.
        """
        return core_schema.any_schema()


class PreparedDocuments(NamedTuple):
    """Prepared documents ready for ChromaDB insertion.

    Attributes:
        ids: List of document IDs
        texts: List of document texts
        metadatas: List of document metadata mappings (empty dict for no metadata)
    """

    ids: list[str]
    texts: list[str]
    metadatas: list[Mapping[str, str | int | float | bool]]


class ExtractedSearchParams(NamedTuple):
    """Extracted search parameters for ChromaDB queries.

    Attributes:
        collection_name: Name of the collection to search
        query: Search query text
        limit: Maximum number of results
        metadata_filter: Optional metadata filter
        score_threshold: Optional minimum similarity score
        where: Optional ChromaDB where clause
        where_document: Optional ChromaDB document filter
        include: Fields to include in results
    """

    collection_name: str
    query: str
    limit: int
    metadata_filter: dict[str, Any] | None
    score_threshold: float | None
    where: Where | None
    where_document: WhereDocument | None
    include: Include


class ChromaDBCollectionCreateParams(BaseCollectionParams, total=False):
    """Parameters for creating a ChromaDB collection.

    This class extends BaseCollectionParams to include any additional
    parameters specific to ChromaDB collection creation.
    """

    configuration: CollectionConfigurationInterface
    metadata: CollectionMetadata
    embedding_function: ChromaEmbeddingFunction
    data_loader: DataLoader[Loadable]
    get_or_create: bool


class ChromaDBCollectionSearchParams(BaseCollectionSearchParams, total=False):
    """Parameters for searching a ChromaDB collection.

    This class extends BaseCollectionSearchParams to include ChromaDB-specific
    search parameters like where clauses and include options.
    """

    where: Where
    where_document: WhereDocument
    include: Include

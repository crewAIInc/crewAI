import hashlib
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

import chromadb
import chromadb.errors
from chromadb.api import ClientAPI
from chromadb.api.types import OneOrMany
from chromadb.config import Settings
import warnings

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.embeddings.configurator import EmbeddingConfigurator
from crewai.utilities.chromadb import sanitize_collection_name
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger
from crewai.utilities.paths import db_storage_path
from crewai.utilities.chromadb import create_persistent_client
from crewai.utilities.logger_utils import suppress_logging


class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    collection: Optional[chromadb.Collection] = None
    collection_name: Optional[str] = "knowledge"
    app: Optional[ClientAPI] = None

    def __init__(
        self,
        embedder: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self._set_embedder_config(embedder)

    def search(
        self,
        query: List[str],
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        with suppress_logging(
            "chromadb.segment.impl.vector.local_persistent_hnsw", logging.ERROR
        ):
            if self.collection:
                fetched = self.collection.query(
                    query_texts=query,
                    n_results=limit,
                    where=filter,
                )
                # Use list comprehension with zip for better performance
                results = [
                    {
                        "id": id_,
                        "metadata": metadata,
                        "context": document,
                        "score": distance,
                    }
                    for id_, metadata, document, distance in zip(
                        fetched["ids"][0],  # type: ignore
                        fetched["metadatas"][0],  # type: ignore
                        fetched["documents"][0],  # type: ignore
                        fetched["distances"][0],  # type: ignore
                    )
                    if distance >= score_threshold
                ]
                return results
            else:
                raise RuntimeError("Collection not initialized")

    def initialize_knowledge_storage(self):
        # Suppress deprecation warnings from chromadb, which are not relevant to us
        # TODO: Remove this once we upgrade chromadb to at least 1.0.8.
        warnings.filterwarnings(
            "ignore",
            message=r".*'model_fields'.*is deprecated.*",
            module=r"^chromadb(\.|$)",
        )

        self.app = create_persistent_client(
            path=os.path.join(db_storage_path(), "knowledge"),
            settings=Settings(allow_reset=True),
        )

        try:
            collection_name = (
                f"knowledge_{self.collection_name}"
                if self.collection_name
                else "knowledge"
            )
            if self.app:
                self.collection = self.app.get_or_create_collection(
                    name=sanitize_collection_name(collection_name),
                    embedding_function=self.embedder,
                )
            else:
                raise RuntimeError("Vector Database Client not initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to create or get collection: {str(e)}") from e

    def reset(self):
        base_path = os.path.join(db_storage_path(), KNOWLEDGE_DIRECTORY)
        if not self.app:
            self.app = create_persistent_client(
                path=base_path, settings=Settings(allow_reset=True)
            )

        self.app.reset()
        shutil.rmtree(base_path)
        self.app = None
        self.collection = None

    def save(
        self,
        documents: List[str],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        if not self.collection:
            raise RuntimeError("Collection not initialized")

        try:
            # Create a dictionary to store unique documents
            unique_docs = {}

            # Prepare metadata list for easier access
            metadata_list: List[Optional[Dict[str, Any]]]
            if metadata is None:
                metadata_list = [None] * len(documents)
            elif isinstance(metadata, list):
                metadata_list = list(
                    metadata
                )  # Create a copy to ensure type compatibility
            else:
                metadata_list = [metadata] * len(documents)

            # Generate IDs and create a mapping of id -> (document, metadata) in one pass
            for doc, doc_metadata in zip(documents, metadata_list):
                doc_id = hashlib.sha256(doc.encode("utf-8")).hexdigest()
                unique_docs[doc_id] = (doc, doc_metadata)

            # Build filtered lists directly from unique_docs
            filtered_ids = list(unique_docs.keys())
            filtered_data = list(unique_docs.values())
            filtered_docs = [doc for doc, _ in filtered_data]
            filtered_metadata = [meta for _, meta in filtered_data]

            # If we have no metadata at all, set it to None
            final_metadata: Optional[OneOrMany[chromadb.Metadata]] = (
                None if all(m is None for m in filtered_metadata) else filtered_metadata
            )

            self.collection.upsert(
                documents=filtered_docs,
                metadatas=final_metadata,
                ids=filtered_ids,
            )
        except chromadb.errors.InvalidDimensionException as e:
            Logger(verbose=True).log(
                "error",
                "Embedding dimension mismatch. This usually happens when mixing different embedding models. Try resetting the collection using `crewai reset-memories -a`",
                "red",
            )
            raise ValueError(
                "Embedding dimension mismatch. Make sure you're using the same embedding model "
                "across all operations with this collection."
                "Try resetting the collection using `crewai reset-memories -a`"
            ) from e
        except Exception as e:
            Logger(verbose=True).log("error", f"Failed to upsert documents: {e}", "red")
            raise

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def _set_embedder_config(self, embedder: Optional[Dict[str, Any]] = None) -> None:
        """Set the embedding configuration for the knowledge storage.

        Args:
            embedder_config (Optional[Dict[str, Any]]): Configuration dictionary for the embedder.
                If None or empty, defaults to the default embedding function.
        """
        self.embedder = (
            EmbeddingConfigurator().configure_embedder(embedder)
            if embedder
            else self._create_default_embedding_function()
        )

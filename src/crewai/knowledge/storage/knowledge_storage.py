import hashlib
import logging
import os
import shutil
import warnings
from collections.abc import Mapping
from typing import Any, Optional, Union

import chromadb
import chromadb.errors
from chromadb import EmbeddingFunction
from chromadb.api import ClientAPI
from chromadb.api.types import OneOrMany
from chromadb.config import Settings

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.embeddings.configurator import EmbeddingConfigurator
from crewai.utilities.chromadb import create_persistent_client, sanitize_collection_name
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger
from crewai.utilities.logger_utils import suppress_logging
from crewai.utilities.paths import db_storage_path


class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    collection: Optional[chromadb.Collection] = None
    collection_name: Optional[str] = "knowledge"
    app: Optional[ClientAPI] = None
    embedder: Optional[EmbeddingFunction[Any]] = None

    def __init__(
        self,
        embedder: Optional[dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self._set_embedder_config(embedder)

    def search(
        self,
        query: list[str],
        limit: int = 3,
        filter: Optional[dict[str, Any]] = None,
        score_threshold: float = 0.35,
    ) -> list[dict[str, Any]]:
        with suppress_logging(
            "chromadb.segment.impl.vector.local_persistent_hnsw", logging.ERROR
        ):
            if self.collection:
                fetched = self.collection.query(
                    query_texts=query,
                    n_results=limit,
                    where=filter,
                )
                results = []
                if (
                    fetched
                    and "ids" in fetched
                    and fetched["ids"]
                    and len(fetched["ids"]) > 0
                ):
                    ids_list = (
                        fetched["ids"][0]
                        if isinstance(fetched["ids"][0], list)
                        else fetched["ids"]
                    )
                    for i in range(len(ids_list)):
                        # Handle metadatas
                        metadata = {}
                        if fetched.get("metadatas") and len(fetched["metadatas"]) > 0:
                            metadata_list = (
                                fetched["metadatas"][0]
                                if isinstance(fetched["metadatas"][0], list)
                                else fetched["metadatas"]
                            )
                            if i < len(metadata_list):
                                metadata = metadata_list[i]

                        # Handle documents
                        context = ""
                        if fetched.get("documents") and len(fetched["documents"]) > 0:
                            docs_list = (
                                fetched["documents"][0]
                                if isinstance(fetched["documents"][0], list)
                                else fetched["documents"]
                            )
                            if i < len(docs_list):
                                context = docs_list[i]

                        # Handle distances
                        score = 1.0
                        if fetched.get("distances") and len(fetched["distances"]) > 0:
                            dist_list = (
                                fetched["distances"][0]
                                if isinstance(fetched["distances"][0], list)
                                else fetched["distances"]
                            )
                            if i < len(dist_list):
                                score = dist_list[i]

                        result = {
                            "id": ids_list[i],
                            "metadata": metadata,
                            "context": context,
                            "score": score,
                        }

                        # Check score threshold - distances are smaller when more similar
                        if isinstance(score, (int, float)) and score <= score_threshold:
                            results.append(result)
                return results
            else:
                raise Exception("Collection not initialized")

    def initialize_knowledge_storage(self) -> None:
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
                raise Exception("Vector Database Client not initialized")
        except Exception:
            raise Exception("Failed to create or get collection")

    def reset(self) -> None:
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
        documents: list[str],
        metadata: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
    ) -> None:
        if not self.collection:
            raise Exception("Collection not initialized")

        try:
            # Create a dictionary to store unique documents
            unique_docs = {}

            # Generate IDs and create a mapping of id -> (document, metadata)
            for idx, doc in enumerate(documents):
                doc_id = hashlib.sha256(doc.encode("utf-8")).hexdigest()
                doc_metadata = None
                if metadata is not None:
                    if isinstance(metadata, list):
                        doc_metadata = metadata[idx]
                    else:
                        doc_metadata = metadata
                unique_docs[doc_id] = (doc, doc_metadata)

            # Prepare filtered lists for ChromaDB
            filtered_docs = []
            filtered_metadata = []
            filtered_ids = []

            # Build the filtered lists
            for doc_id, (doc, meta) in unique_docs.items():
                filtered_docs.append(doc)
                filtered_metadata.append(meta)
                filtered_ids.append(doc_id)

            # If we have no metadata at all, set it to None
            final_metadata: Optional[OneOrMany[chromadb.Metadata]] = (
                None if all(m is None for m in filtered_metadata) else filtered_metadata  # type: ignore[assignment]
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

    def _create_default_embedding_function(self) -> Any:
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def _set_embedder_config(self, embedder: Optional[dict[str, Any]] = None) -> None:
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

import contextlib
import hashlib
import io
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union

try:
    import chromadb
    import chromadb.errors
    from chromadb.api import ClientAPI
    from chromadb.api.types import OneOrMany
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None  # type: ignore
    ClientAPI = Any  # type: ignore
    OneOrMany = Any  # type: ignore
    Settings = Any  # type: ignore
    HAS_CHROMADB = False

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.chromadb import sanitize_collection_name
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.errors import ChromaDBRequiredError
from crewai.utilities.logger import Logger
from crewai.utilities.paths import db_storage_path


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)


class KnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    collection: Optional[Any] = None  # type: ignore
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
        if not HAS_CHROMADB:
            raise ChromaDBRequiredError("knowledge storage")
            
        with suppress_logging():
            if self.collection:
                fetched = self.collection.query(
                    query_texts=query,
                    n_results=limit,
                    where=filter,
                )
                results = []
                for i in range(len(fetched["ids"][0])):  # type: ignore
                    result = {
                        "id": fetched["ids"][0][i],  # type: ignore
                        "metadata": fetched["metadatas"][0][i],  # type: ignore
                        "context": fetched["documents"][0][i],  # type: ignore
                        "score": fetched["distances"][0][i],  # type: ignore
                    }
                    if result["score"] >= score_threshold:
                        results.append(result)
                return results
            else:
                raise Exception("Collection not initialized")

    def initialize_knowledge_storage(self):
        if not HAS_CHROMADB:
            raise ChromaDBRequiredError("knowledge storage")
            
        base_path = os.path.join(db_storage_path(), "knowledge")
        try:
            chroma_client = chromadb.PersistentClient(
                path=base_path,
                settings=Settings(allow_reset=True),
            )

            self.app = chroma_client

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
        except ImportError:
            raise ChromaDBRequiredError("knowledge storage")

    def reset(self):
        if not HAS_CHROMADB:
            raise ChromaDBRequiredError("knowledge storage")
            
        base_path = os.path.join(db_storage_path(), KNOWLEDGE_DIRECTORY)
        try:
            if not self.app:
                self.app = chromadb.PersistentClient(
                    path=base_path,
                    settings=Settings(allow_reset=True),
                )

            self.app.reset()
            shutil.rmtree(base_path)
            self.app = None
            self.collection = None
        except ImportError:
            raise ChromaDBRequiredError("knowledge storage")

    def save(
        self,
        documents: List[str],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ):
        if not HAS_CHROMADB:
            raise ChromaDBRequiredError("knowledge storage")
            
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
            final_metadata: Optional[OneOrMany[Any]] = (
                None if all(m is None for m in filtered_metadata) else filtered_metadata
            )

            self.collection.upsert(
                documents=filtered_docs,
                metadatas=final_metadata,
                ids=filtered_ids,
            )
        except ImportError:
            raise ChromaDBRequiredError("knowledge storage")
        except Exception as e:
            if HAS_CHROMADB and isinstance(e, chromadb.errors.InvalidDimensionException):
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
            else:
                Logger(verbose=True).log("error", f"Failed to upsert documents: {e}", "red")
                raise

    def _create_default_embedding_function(self):
        if not HAS_CHROMADB:
            raise ChromaDBRequiredError("knowledge storage")
            
        try:
            from chromadb.utils.embedding_functions.openai_embedding_function import (
                OpenAIEmbeddingFunction,
            )

            return OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
            )
        except ImportError:
            raise ChromaDBRequiredError("knowledge storage")

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

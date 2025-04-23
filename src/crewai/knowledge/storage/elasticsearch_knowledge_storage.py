import contextlib
import hashlib
import io
import logging
import os
from typing import Any, Dict, List, Optional, Union, cast

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.logger import Logger
from crewai.utilities.paths import db_storage_path


@contextlib.contextmanager
def suppress_logging(logger_name="elasticsearch", level=logging.ERROR):
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


class ElasticsearchKnowledgeStorage(BaseKnowledgeStorage):
    """
    Extends BaseKnowledgeStorage to use Elasticsearch for storing embeddings
    and improving search efficiency.
    """

    app: Any = None
    collection_name: Optional[str] = "knowledge"

    def __init__(
        self,
        embedder_config: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        host: str = "localhost",
        port: int = 9200,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs: Any
    ):
        self.collection_name = collection_name
        self._set_embedder_config(embedder_config)
        
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index_name = f"crewai_knowledge_{collection_name if collection_name else 'default'}".lower()
        self.additional_config = kwargs

    def search(
        self,
        query: List[str],
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        if not self.app:
            self.initialize_knowledge_storage()
            
        try:
            embedding = self._get_embedding_for_text(query[0])
            
            search_query: Dict[str, Any] = {
                "size": limit,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": embedding}
                        }
                    }
                }
            }
            
            if filter:
                query_obj = search_query.get("query", {})
                if isinstance(query_obj, dict):
                    script_score_obj = query_obj.get("script_score", {})
                    if isinstance(script_score_obj, dict):
                        query_part = script_score_obj.get("query", {})
                        if isinstance(query_part, dict):
                            for key, value in filter.items():
                                script_score_obj["query"] = {
                                    "bool": {
                                        "must": [
                                            query_part,
                                            {"match": {f"metadata.{key}": value}}
                                        ]
                                    }
                                }
            
            with suppress_logging():
                if self.app is not None and hasattr(self.app, "search") and callable(getattr(self.app, "search")):
                    response = self.app.search(
                        index=self.index_name,
                        body=search_query
                    )
                
                    results = []
                    for hit in response["hits"]["hits"]:
                        adjusted_score = (hit["_score"] - 1.0)
                        
                        if adjusted_score >= score_threshold:
                            results.append({
                                "id": hit["_id"],
                                "metadata": hit["_source"]["metadata"],
                                "context": hit["_source"]["text"],
                                "score": adjusted_score,
                            })
                        
                    return results
                else:
                    Logger(verbose=True).log("error", "Elasticsearch client is not initialized", "red")
                    return []
        except Exception as e:
            Logger(verbose=True).log("error", f"Search error: {e}", "red")
            raise Exception(f"Error during knowledge search: {str(e)}")

    def initialize_knowledge_storage(self):
        try:
            from elasticsearch import Elasticsearch
            
            es_auth = {}
            if self.username and self.password:
                es_auth = {"basic_auth": (self.username, self.password)}
                
            self.app = Elasticsearch(
                [f"http://{self.host}:{self.port}"], 
                **es_auth,
                **self.additional_config
            )
            
            if not self.app.indices.exists(index=self.index_name):
                self.app.indices.create(
                    index=self.index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "text": {"type": "text"},
                                "embedding": {
                                    "type": "dense_vector",
                                    "dims": 1536,  # Default for OpenAI embeddings
                                    "index": True,
                                    "similarity": "cosine"
                                },
                                "metadata": {"type": "object"}
                            }
                        }
                    }
                )
                
        except ImportError:
            raise ImportError(
                "Elasticsearch is not installed. Please install it with `pip install elasticsearch`."
            )
        except Exception as e:
            Logger(verbose=True).log(
                "error", 
                f"Error initializing Elasticsearch: {str(e)}", 
                "red"
            )
            raise Exception(f"Error initializing Elasticsearch: {str(e)}")

    def reset(self) -> None:
        try:
            if self.app is not None:
                if self.app.indices.exists(index=self.index_name):
                    self.app.indices.delete(index=self.index_name)
                
                self.initialize_knowledge_storage()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the knowledge storage: {e}"
            )
               
    def save(
        self,
        documents: List[str],
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> None:
        if not self.app:
            self.initialize_knowledge_storage()

        try:
            unique_docs = {}

            for idx, doc in enumerate(documents):
                doc_id = hashlib.sha256(doc.encode("utf-8")).hexdigest()
                doc_metadata = None
                if metadata is not None:
                    if isinstance(metadata, list):
                        doc_metadata = metadata[idx]
                    else:
                        doc_metadata = metadata
                unique_docs[doc_id] = (doc, doc_metadata)

            for doc_id, (doc, meta) in unique_docs.items():
                embedding = self._get_embedding_for_text(doc)
                
                doc_body = {
                    "text": doc,
                    "embedding": embedding,
                    "metadata": meta or {},
                }
                
                if self.app is not None and hasattr(self.app, "index") and callable(getattr(self.app, "index")):
                    self.app.index(
                        index=self.index_name,
                        id=doc_id,
                        document=doc_body,
                        refresh=True  # Make the document immediately available for search
                    )
                else:
                    Logger(verbose=True).log("error", "Elasticsearch client is not initialized", "red")
                
        except Exception as e:
            Logger(verbose=True).log("error", f"Save error: {e}", "red")
            raise Exception(f"Error during knowledge save: {str(e)}")

    def _get_embedding_for_text(self, text: str) -> List[float]:
        """Get embedding for text using the configured embedder."""
        if self.embedder_config is None:
            raise ValueError("Embedder configuration is not set")
            
        embedder = self.embedder_config
        if hasattr(embedder, "embed_documents") and callable(getattr(embedder, "embed_documents")):
            return embedder.embed_documents([text])[0]
        elif hasattr(embedder, "embed") and callable(getattr(embedder, "embed")):
            return embedder.embed(text)
        else:
            raise ValueError("Invalid embedding function configuration")

    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

    def _set_embedder_config(
        self, embedder: Optional[Dict[str, Any]] = None
    ) -> None:
        """Set the embedding configuration for the knowledge storage.

        Args:
            embedder (Optional[Dict[str, Any]]): Configuration dictionary for the embedder.
                If None or empty, defaults to the default embedding function.
        """
        self.embedder_config = (
            EmbeddingConfigurator().configure_embedder(embedder)
            if embedder
            else self._create_default_embedding_function()
        )

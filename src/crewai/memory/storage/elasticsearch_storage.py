import contextlib
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.constants import MAX_FILE_NAME_LENGTH
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


class ElasticsearchStorage(BaseRAGStorage):
    """
    Extends BaseRAGStorage to use Elasticsearch for storing embeddings
    and improving search efficiency.
    """

    app: Any | None = None

    def __init__(
        self, 
        type, 
        allow_reset=True, 
        embedder_config=None, 
        crew=None, 
        path=None,
        host="localhost",
        port=9200,
        username=None,
        password=None,
        **kwargs
    ):
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents
        self.storage_file_name = self._build_storage_file_name(type, agents)
        
        self.type = type
        self.allow_reset = allow_reset
        self.path = path
        
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index_name = f"crewai_{type}".lower()
        self.additional_config = kwargs
        
        self._initialize_app()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory and index names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def _build_storage_file_name(self, type: str, file_name: str) -> str:
        """
        Ensures file name does not exceed max allowed by OS
        """
        base_path = f"{db_storage_path()}/{type}"

        if len(file_name) > MAX_FILE_NAME_LENGTH:
            logging.warning(
                f"Trimming file name from {len(file_name)} to {MAX_FILE_NAME_LENGTH} characters."
            )
            file_name = file_name[:MAX_FILE_NAME_LENGTH]

        return f"{base_path}/{file_name}"

    def _set_embedder_config(self):
        configurator = EmbeddingConfigurator()
        self.embedder_config = configurator.configure_embedder(self.embedder_config)

    def _initialize_app(self):
        try:
            from elasticsearch import Elasticsearch
            
            self._set_embedder_config()
            
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

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app"):
            self._initialize_app()
            
        try:
            self._generate_embedding(value, metadata)
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app") or self.app is None:
            self._initialize_app()

        try:
            embedding = self._get_embedding_for_text(query)
            
            search_query = {
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
                for key, value in filter.items():
                    search_query["query"]["script_score"]["query"] = {
                        "bool": {
                            "must": [
                                search_query["query"]["script_score"]["query"],
                                {"match": {f"metadata.{key}": value}}
                            ]
                        }
                    }
            
            with suppress_logging():
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
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _get_embedding_for_text(self, text: str) -> List[float]:
        """Get embedding for text using the configured embedder."""
        if hasattr(self.embedder_config, "embed_documents"):
            return self.embedder_config.embed_documents([text])[0]
        elif hasattr(self.embedder_config, "embed"):
            return self.embedder_config.embed(text)
        else:
            raise ValueError("Invalid embedding function configuration")

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app") or self.app is None:
            self._initialize_app()

        embedding = self._get_embedding_for_text(text)
        
        doc = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        
        self.app.index(
            index=self.index_name,
            id=str(uuid.uuid4()),
            document=doc,
            refresh=True  # Make the document immediately available for search
        )

    def reset(self) -> None:
        try:
            if self.app:
                if self.app.indices.exists(index=self.index_name):
                    self.app.indices.delete(index=self.index_name)
                
                self._initialize_app()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )

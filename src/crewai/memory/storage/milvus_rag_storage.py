import logging
import os
import uuid
from typing import Any, Dict, List, Optional


from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities import EmbeddingConfigurator
from crewai.utilities.constants import MAX_FILE_NAME_LENGTH
from crewai.utilities.paths import db_storage_path

DEFAULT_EMBEDDING_DIM = 1536

class MilvusRAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries using Milvus.
    """

    def __init__(
        self,
        type,
        allow_reset=True,
        embedder_config=None,
        crew=None,
        milvus_config=None,
    ):
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents
        self.storage_file_name = self._build_storage_file_name(type, agents)
        
        self.type = type
        self.allow_reset = allow_reset
        self.milvus_config = milvus_config or {"uri": "./milvus.db", "token": ""}
        
        # Use agents in collection name to ensure unique collections per agent combination
        self.collection_name = f"{type}_{agents}" if agents else f"{type}_collection"
            
        self._initialize_app()

    def _set_embedder_config(self):
        configurator = EmbeddingConfigurator()
        if self.embedder_config:
            self.embedding_function = configurator.configure_embedder(self.embedder_config)


    def _initialize_app(self):
        from pymilvus import MilvusClient
        self._set_embedder_config()
        self.client = MilvusClient(uri=self.milvus_config.get("uri", ""), token=self.milvus_config.get("token", ""))

        # Always drop the collection if it exists and create a new one
        try:
            collections = self.client.list_collections()
            if self.collection_name in collections:
                logging.info(f"Dropping existing collection: {self.collection_name}")
                self.client.drop_collection(self.collection_name)
            
            # Create a new collection
            self._create_collection()
        except Exception as e:
            logging.error(f"Error initializing Milvus: {str(e)}")
            # Try to create collection anyway
            try:
                self._create_collection()
            except Exception as inner_e:
                logging.error(f"Failed to create collection: {str(inner_e)}")

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
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

    def _create_collection(self):
        from pymilvus import DataType
        
        try:
            logging.info(f"Creating new collection: {self.collection_name}")
            
            schema = self.client.create_schema(
                auto_id=False,
                enable_dynamic_field=False,
            )
            
            schema.add_field(
                field_name="id", 
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=65535
            )
            schema.add_field(
                field_name="embedding", 
                datatype=DataType.FLOAT_VECTOR, 
                dim=DEFAULT_EMBEDDING_DIM
            )
            schema.add_field(
                field_name="text", 
                datatype=DataType.VARCHAR,
                max_length=65535
            )
            schema.add_field(
                field_name="metadata", 
                datatype=DataType.JSON
            )
            
            index_params = self.client.prepare_index_params()
            
            index_params.add_index(
                field_name="embedding", 
                index_type="AUTOINDEX",
                metric_type="L2",
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            logging.info(f"Successfully created collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error creating collection: {str(e)}")
            raise RuntimeError(f"Failed to create Milvus collection: {str(e)}")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        try:
            embedding = self._generate_embedding(value, metadata)
            
            data = [{
                "id": str(uuid.uuid4()),
                "embedding": embedding,
                "text": value,
                "metadata": metadata
            }]
            
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            logging.info(f"Successfully saved data to collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        try:
            embedding = self._generate_embedding(query, {})
            search_params = {
                "metric_type": "L2", 
                "params": {"nprobe": 128},
            }
            
            search_args = {
                "collection_name": self.collection_name,
                "data": [embedding],  
                "limit": limit,      
                "output_fields": ["text", "metadata"],  
                "search_params": search_params  
            }

            results = self.client.search(**search_args)
            processed_results = []
            for result_group in results:
                for hit in result_group:
                    if hit['distance'] <= score_threshold:
                        entity = hit.get('entity', {})
                        processed_results.append({
                            "id": hit['id'],
                            "metadata": entity.get("metadata", {}),
                            "context": entity.get("text", ""),
                            "score": hit['distance'],
                        })
            
            logging.info(f"Search returned {len(processed_results)} results")
            return processed_results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _generate_embedding(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[float]:
        """Generate embedding using the configured embedder."""
        if not hasattr(self, 'embedder_config') or not self.embedder_config:
            from pymilvus import model
            embedding_function = model.dense.OpenAIEmbeddingFunction(
                model_name='text-embedding-ada-002', 
                api_key=os.getenv("OPENAI_API_KEY"), 
            )
            logging.info("Embedder configuration is missing. Using Milvus OpenAI Embedding Function")
            return embedding_function.encode_queries([text])[0]

        # Use the configured embedding function
        else:
            embedding =  self.embedding_function([text])
            return embedding[0]


    def reset(self) -> None:
        try:
            collections = self.client.list_collections()
            if self.collection_name in collections:
                logging.info(f"Resetting collection: {self.collection_name}")
                self.client.drop_collection(self.collection_name)
                self._create_collection()
        except Exception as e:
            logging.error(f"An error occurred while resetting the {self.type} memory: {e}")

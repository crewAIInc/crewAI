import json
import os
from importlib.metadata import version
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from openai import AzureOpenAI, Client
from pydantic import BaseModel, Field

from crewai_tools.tools.mongodb_vector_search_tool.utils import (
    create_vector_search_index,
)

try:
    import pymongo  # noqa: F403

    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = getLogger(__name__)


class MongoDBVectorSearchConfig(BaseModel):
    """Configuration for MongoDB vector search queries."""

    limit: Optional[int] = Field(
        default=4, description="number of documents to return."
    )
    pre_filter: Optional[dict[str, Any]] = Field(
        default=None,
        description="List of MQL match expressions comparing an indexed field",
    )
    post_filter_pipeline: Optional[list[dict]] = Field(
        default=None,
        description="Pipeline of MongoDB aggregation stages to filter/process results after $vectorSearch.",
    )
    oversampling_factor: int = Field(
        default=10,
        description="Multiple of limit used when generating number of candidates at each step in the HNSW Vector Search",
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include the embedding vector of each result in metadata.",
    )


class MongoDBToolSchema(MongoDBVectorSearchConfig):
    """Input for MongoDBTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the MongoDB database. Pass only the query, not the question.",
    )


class MongoDBVectorSearchTool(BaseTool):
    """Tool to perfrom a vector search the MongoDB database"""

    name: str = "MongoDBVectorSearchTool"
    description: str = "A tool to perfrom a vector search on a MongoDB database for relevant information on internal documents."

    args_schema: Type[BaseModel] = MongoDBToolSchema
    query_config: Optional[MongoDBVectorSearchConfig] = Field(
        default=None, description="MongoDB Vector Search query configuration"
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Text OpenAI embedding model to use",
    )
    vector_index_name: str = Field(
        default="vector_index", description="Name of the Atlas Search vector index"
    )
    text_key: str = Field(
        default="text",
        description="MongoDB field that will contain the text for each document",
    )
    embedding_key: str = Field(
        default="embedding",
        description="Field that will contain the embedding for each document",
    )
    database_name: str = Field(..., description="The name of the MongoDB database")
    collection_name: str = Field(..., description="The name of the MongoDB collection")
    connection_string: str = Field(
        ...,
        description="The connection string of the MongoDB cluster",
    )
    dimensions: int = Field(
        default=1536,
        description="Number of dimensions in the embedding vector",
    )
    env_vars: List[EnvVar] = [
        EnvVar(
            name="BROWSERBASE_API_KEY",
            description="API key for Browserbase services",
            required=False,
        ),
        EnvVar(
            name="BROWSERBASE_PROJECT_ID",
            description="Project ID for Browserbase services",
            required=False,
        ),
    ]
    package_dependencies: List[str] = ["mongdb"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not MONGODB_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'mongodb' crewai tool. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "pymongo"], check=True)

            else:
                raise ImportError("You are missing the 'mongodb' crewai tool.")

        if "AZURE_OPENAI_ENDPOINT" in os.environ:
            self._openai_client = AzureOpenAI()
        elif "OPENAI_API_KEY" in os.environ:
            self._openai_client = Client()
        else:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for MongoDBVectorSearchTool and it is mandatory to use the tool."
            )

        from pymongo import MongoClient
        from pymongo.driver_info import DriverInfo

        self._client = MongoClient(
            self.connection_string,
            driver=DriverInfo(name="CrewAI", version=version("crewai-tools")),
        )
        self._coll = self._client[self.database_name][self.collection_name]

    def create_vector_search_index(
        self,
        *,
        dimensions: int,
        relevance_score_fn: str = "cosine",
        auto_index_timeout: int = 15,
    ) -> None:
        """Convenience function to create a vector search index.

        Args:
            dimensions: Number of dimensions in embedding.  If the value is set and
                the index does not exist, an index will be created.
            relevance_score_fn: The similarity score used for the index
                Currently supported: 'euclidean', 'cosine', and 'dotProduct'
            auto_index_timeout: Timeout in seconds to wait for an auto-created index
               to be ready.
        """

        create_vector_search_index(
            collection=self._coll,
            index_name=self.vector_index_name,
            dimensions=dimensions,
            path=self.embedding_key,
            similarity=relevance_score_fn,
            wait_until_complete=auto_index_timeout,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts, create embeddings, and add to the Collection and index.

        Important notes on ids:
            - If _id or id is a key in the metadatas dicts, one must
                pop them and provide as separate list.
            - They must be unique.
            - If they are not provided, the VectorStore will create unique ones,
                stored as bson.ObjectIds internally, and strings in Langchain.
                These will appear in Document.metadata with key, '_id'.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique ids that will be used as index in VectorStore.
                See note on ids.
            batch_size: Number of documents to insert at a time.
                Tuning this may help with performance and sidestep MongoDB limits.

        Returns:
            List of ids added to the vectorstore.
        """
        from bson import ObjectId

        _metadatas = metadatas or [{} for _ in texts]
        ids = [str(ObjectId()) for _ in range(len(list(texts)))]
        metadatas_batch = _metadatas

        result_ids = []
        texts_batch = []
        metadatas_batch = []
        size = 0
        i = 0
        for j, (text, metadata) in enumerate(zip(texts, _metadatas)):
            size += len(text) + len(metadata)
            texts_batch.append(text)
            metadatas_batch.append(metadata)
            if (j + 1) % batch_size == 0 or size >= 47_000_000:
                batch_res = self._bulk_embed_and_insert_texts(
                    texts_batch, metadatas_batch, ids[i : j + 1]
                )
                result_ids.extend(batch_res)
                texts_batch = []
                metadatas_batch = []
                size = 0
                i = j + 1
        if texts_batch:
            batch_res = self._bulk_embed_and_insert_texts(
                texts_batch, metadatas_batch, ids[i : j + 1]
            )
            result_ids.extend(batch_res)
        return result_ids

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [
            i.embedding
            for i in self._openai_client.embeddings.create(
                input=texts,
                model=self.embedding_model,
                dimensions=self.dimensions,
            ).data
        ]

    def _bulk_embed_and_insert_texts(
        self,
        texts: List[str],
        metadatas: List[dict],
        ids: List[str],
    ) -> List[str]:
        """Bulk insert single batch of texts, embeddings, and ids."""
        from bson import ObjectId
        from pymongo.operations import ReplaceOne

        if not texts:
            return []
        # Compute embedding vectors
        embeddings = self._embed_texts(texts)
        docs = [
            {
                "_id": ObjectId(i),
                self.text_key: t,
                self.embedding_key: embedding,
                **m,
            }
            for i, t, m, embedding in zip(ids, texts, metadatas, embeddings)
        ]
        operations = [ReplaceOne({"_id": doc["_id"]}, doc, upsert=True) for doc in docs]
        # insert the documents in MongoDB Atlas
        result = self._coll.bulk_write(operations)
        assert result.upserted_ids is not None
        return [str(_id) for _id in result.upserted_ids.values()]

    def _run(self, query: str) -> str:
        try:
            query_config = self.query_config or MongoDBVectorSearchConfig()
            limit = query_config.limit
            oversampling_factor = query_config.oversampling_factor
            pre_filter = query_config.pre_filter
            include_embeddings = query_config.include_embeddings
            post_filter_pipeline = query_config.post_filter_pipeline

            # Create the embedding for the query
            query_vector = self._embed_texts([query])[0]

            # Atlas Vector Search, potentially with filter
            stage = {
                "index": self.vector_index_name,
                "path": self.embedding_key,
                "queryVector": query_vector,
                "numCandidates": limit * oversampling_factor,
                "limit": limit,
            }
            if pre_filter:
                stage["filter"] = pre_filter

            pipeline = [
                {"$vectorSearch": stage},
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            ]

            # Remove embeddings unless requested
            if not include_embeddings:
                pipeline.append({"$project": {self.embedding_key: 0}})

            # Post-processing
            if post_filter_pipeline is not None:
                pipeline.extend(post_filter_pipeline)

            # Execution
            cursor = self._coll.aggregate(pipeline)  # type: ignore[arg-type]
            docs = []

            # Format
            for doc in cursor:
                docs.append(doc)
            return json.dumps(docs)
        except Exception as e:
            logger.error(f"Error: {e}")
            return ""

    def __del__(self):
        """Cleanup clients on deletion."""
        try:
            if hasattr(self, "_client") and self._client:
                self._client.close()
        except Exception as e:
            logger.error(f"Error: {e}")

        try:
            if hasattr(self, "_openai_client") and self._openai_client:
                self._openai_client.close()
        except Exception as e:
            logger.error(f"Error: {e}")

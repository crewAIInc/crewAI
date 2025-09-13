import json
import os
from typing import Any, Optional, Type, List, Dict, Callable

try:
    import couchbase.search as search
    from couchbase.cluster import Cluster
    from couchbase.options import SearchOptions
    from couchbase.vector_search import VectorQuery, VectorSearch

    COUCHBASE_AVAILABLE = True
except ImportError:
    COUCHBASE_AVAILABLE = False
    search = Any
    Cluster = Any
    SearchOptions = Any
    VectorQuery = Any
    VectorSearch = Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, SkipValidation


class CouchbaseToolSchema(BaseModel):
    """Input for CouchbaseTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Couchbase database. Pass only the query, not the question.",
    )

class CouchbaseFTSVectorSearchTool(BaseTool):
    """Tool to search the Couchbase database"""

    model_config = {"arbitrary_types_allowed": True}
    name: str = "CouchbaseFTSVectorSearchTool"
    description: str = "A tool to search the Couchbase database for relevant information on internal documents."
    args_schema: Type[BaseModel] = CouchbaseToolSchema
    cluster: SkipValidation[Optional[Cluster]] = None
    collection_name: Optional[str] = None,
    scope_name: Optional[str] = None,
    bucket_name: Optional[str] = None,
    index_name: Optional[str] = None,
    embedding_key: Optional[str] = Field(
        default="embedding",
        description="Name of the field in the search index that stores the vector"
    )
    scoped_index: Optional[bool] = Field(
        default=True,
        description="Specify whether the index is scoped. Is True by default."
    ),
    limit: Optional[int] = Field(default=3)
    embedding_function: SkipValidation[Callable[[str], List[float]]] = Field(
        default=None,
        description="A function that takes a string and returns a list of floats. This is used to embed the query before searching the database."
    )

    def _check_bucket_exists(self) -> bool:
        """Check if the bucket exists in the linked Couchbase cluster"""
        bucket_manager = self.cluster.buckets()
        try:
            bucket_manager.get_bucket(self.bucket_name)
            return True
        except Exception:
            return False

    def _check_scope_and_collection_exists(self) -> bool:
        """Check if the scope and collection exists in the linked Couchbase bucket
        Raises a ValueError if either is not found"""
        scope_collection_map: Dict[str, Any] = {}

        # Get a list of all scopes in the bucket
        for scope in self._bucket.collections().get_all_scopes():
            scope_collection_map[scope.name] = []

            # Get a list of all the collections in the scope
            for collection in scope.collections:
                scope_collection_map[scope.name].append(collection.name)

        # Check if the scope exists
        if self.scope_name not in scope_collection_map.keys():
            raise ValueError(
                f"Scope {self.scope_name} not found in Couchbase "
                f"bucket {self.bucket_name}"
            )

        # Check if the collection exists in the scope
        if self.collection_name not in scope_collection_map[self.scope_name]:
            raise ValueError(
                f"Collection {self.collection_name} not found in scope "
                f"{self.scope_name} in Couchbase bucket {self.bucket_name}"
            )

        return True

    def _check_index_exists(self) -> bool:
        """Check if the Search index exists in the linked Couchbase cluster
        Raises a ValueError if the index does not exist"""
        if self.scoped_index:
            all_indexes = [
                index.name for index in self._scope.search_indexes().get_all_indexes()
            ]
            if self.index_name not in all_indexes:
                raise ValueError(
                    f"Index {self.index_name} does not exist. "
                    " Please create the index before searching."
                )
        else:
            all_indexes = [
                index.name for index in self.cluster.search_indexes().get_all_indexes()
            ]
            if self.index_name not in all_indexes:
                raise ValueError(
                    f"Index {self.index_name} does not exist. "
                    " Please create the index before searching."
                )

        return True

    def __init__(self, **kwargs):
        """Initialize the CouchbaseFTSVectorSearchTool.

        Args:
            **kwargs: Keyword arguments to pass to the BaseTool constructor and
                      to configure the Couchbase connection and search parameters.
                      Requires 'cluster', 'bucket_name', 'scope_name',
                      'collection_name', 'index_name', and 'embedding_function'.

        Raises:
            ValueError: If required parameters are missing, the Couchbase cluster
                        cannot be reached, or the specified bucket, scope,
                        collection, or index does not exist.
        """
        super().__init__(**kwargs)
        if COUCHBASE_AVAILABLE:
            try:
                if not self.cluster:
                    raise ValueError("Cluster instance must be provided")

                if not self.bucket_name:
                    raise ValueError("Bucket name must be provided")

                if not self.scope_name:
                    raise ValueError("Scope name must be provided")

                if not self.collection_name:
                    raise ValueError("Collection name must be provided")

                if not self.index_name:
                    raise ValueError("Index name must be provided")

                if not self.embedding_function:
                    raise ValueError("Embedding function must be provided")

                self._bucket = self.cluster.bucket(self.bucket_name)
                self._scope = self._bucket.scope(self.scope_name)
                self._collection = self._scope.collection(self.collection_name)
            except Exception as e:
                raise ValueError(
                    "Error connecting to couchbase. "
                    "Please check the connection and credentials"
                ) from e

            # check if bucket exists
            if not self._check_bucket_exists():
                raise ValueError(
                    f"Bucket {self.bucket_name} does not exist. "
                    " Please create the bucket before searching."
                )

            self._check_scope_and_collection_exists()
            self._check_index_exists()
        else:
            import click

            if click.confirm(
                "The 'couchbase' package is required to use the CouchbaseFTSVectorSearchTool. "
                "Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "couchbase"], check=True)
            else:
                raise ImportError(
                    "The 'couchbase' package is required to use the CouchbaseFTSVectorSearchTool. "
                    "Please install it with: uv add couchbase"
                )

    def _run(self, query: str) -> str:
        """Execute a vector search query against the Couchbase index.

        Args:
            query: The search query string.

        Returns:
            A JSON string containing the search results.

        Raises:
            ValueError: If the search query fails or returns results without fields.
        """
        query_embedding = self.embedding_function(query)
        fields = ["*"]

        search_req = search.SearchRequest.create(
            VectorSearch.from_vector_query(
                VectorQuery(
                    self.embedding_key,
                    query_embedding,
                    self.limit
                )
            )
        )

        try:
            if self.scoped_index:
                search_iter = self._scope.search(
                    self.index_name,
                    search_req,
                    SearchOptions(
                        limit=self.limit,
                        fields=fields,
                    )
                )
            else:
                search_iter = self.cluster.search(
                    self.index_name,
                    search_req,
                    SearchOptions(
                        limit=self.limit,
                        fields=fields
                    )
                )

            json_response = []

            for row in search_iter.rows():
                json_response.append(row.fields)
        except Exception as e:
            return f"Search failed with error: {e}"

        return json.dumps(json_response, indent=2)
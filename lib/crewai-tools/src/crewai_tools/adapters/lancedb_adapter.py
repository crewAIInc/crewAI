from collections.abc import Callable
from pathlib import Path
from typing import Any

from lancedb import (  # type: ignore[import-untyped]
    DBConnection as LanceDBConnection,
    connect as lancedb_connect,
)
from lancedb.table import Table as LanceDBTable  # type: ignore[import-untyped]
from openai import Client as OpenAIClient
from pydantic import Field, PrivateAttr

from crewai_tools.tools.rag.rag_tool import Adapter


def _default_embedding_function():
    """Create a default embedding function using OpenAI's text-embedding-ada-002 model.

    This function creates and returns an embedding function that uses OpenAI's API
    to generate embeddings for text inputs. The embedding function is used by the
    LanceDBAdapter to convert text queries into vector representations for similarity search.

    Returns:
        Callable: A function that takes a list of strings and returns their embeddings
            as a list of vectors.

    Example:
        >>> embed_fn = _default_embedding_function()
        >>> embeddings = embed_fn(["Hello world"])
        >>> len(embeddings[0])  # Vector dimension
        1536
    """
    client = OpenAIClient()

    def _embedding_function(input):
        rs = client.embeddings.create(input=input, model="text-embedding-ada-002")
        return [record.embedding for record in rs.data]

    return _embedding_function


class LanceDBAdapter(Adapter):
    """Adapter for integrating LanceDB vector database with CrewAI RAG tools.

    LanceDBAdapter provides a bridge between CrewAI's RAG (Retrieval-Augmented Generation)
    system and LanceDB, enabling efficient vector similarity search for knowledge retrieval.
    It handles embedding generation, vector search, and data ingestion with precise control
    over query parameters and column mappings.

    Attributes:
        uri: Database connection URI or path to the LanceDB database.
        table_name: Name of the table to query within the LanceDB database.
        embedding_function: Function to convert text into embeddings. Defaults to OpenAI's
            text-embedding-ada-002 model.
        top_k: Number of top results to return from similarity search. Defaults to 3.
        vector_column_name: Name of the column containing vector embeddings. Defaults to "vector".
        text_column_name: Name of the column containing text content. Defaults to "text".

    Example:
        >>> from crewai_tools.adapters.lancedb_adapter import LanceDBAdapter
        >>> adapter = LanceDBAdapter(
        ...     uri="./my_lancedb",
        ...     table_name="documents",
        ...     top_k=5
        ... )
        >>> results = adapter.query("What is machine learning?")
        >>> print(results)
    """
    uri: str | Path
    table_name: str
    embedding_function: Callable = Field(default_factory=_default_embedding_function)
    top_k: int = 3
    vector_column_name: str = "vector"
    text_column_name: str = "text"

    _db: LanceDBConnection = PrivateAttr()
    _table: LanceDBTable = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the database connection and table after model instantiation.

        This method is automatically called after the Pydantic model is initialized.
        It establishes the connection to the LanceDB database and opens the specified
        table for querying and data operations.

        Args:
            __context: Pydantic context object passed during initialization.

        Raises:
            Exception: If the database connection fails or the table does not exist.
        """
        self._db = lancedb_connect(self.uri)
        self._table = self._db.open_table(self.table_name)

        super().model_post_init(__context)

    def query(self, question: str) -> str:  # type: ignore[override]
        """Perform a vector similarity search for the given question.

        This method converts the input question into an embedding vector and searches
        the LanceDB table for the most similar entries. It returns the top-k results
        based on vector similarity, providing precise retrieval for RAG applications.

        Args:
            question: The text query to search for in the vector database.

        Returns:
            A string containing the concatenated text results from the top-k most
            similar entries, separated by newlines.

        Example:
            >>> adapter = LanceDBAdapter(uri="./db", table_name="docs")
            >>> results = adapter.query("What is CrewAI?")
            >>> print(results)
            CrewAI is a framework for orchestrating AI agents...
            CrewAI provides precise control over agent workflows...
        """
        query = self.embedding_function([question])[0]
        results = (
            self._table.search(query, vector_column_name=self.vector_column_name)
            .limit(self.top_k)
            .select([self.text_column_name])
            .to_list()
        )
        values = [result[self.text_column_name] for result in results]
        return "\n".join(values)

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add data to the LanceDB table.

        This method provides a direct interface to add new records to the underlying
        LanceDB table. It accepts the same arguments as the LanceDB table's add method,
        allowing flexible data ingestion for building knowledge bases.

        Args:
            *args: Positional arguments to pass to the LanceDB table's add method.
            **kwargs: Keyword arguments to pass to the LanceDB table's add method.
                Common kwargs include 'data' (list of records) and 'mode' (append/overwrite).

        Example:
            >>> adapter = LanceDBAdapter(uri="./db", table_name="docs")
            >>> data = [
            ...     {"text": "CrewAI enables agent collaboration", "vector": [0.1, 0.2, ...]},
            ...     {"text": "LanceDB provides vector storage", "vector": [0.3, 0.4, ...]}
            ... ]
            >>> adapter.add(data=data)
        """
        self._table.add(*args, **kwargs)

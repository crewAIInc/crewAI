from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

from crewai.utilities.lock_store import lock as store_lock
from lancedb import (  # type: ignore[import-untyped]
    connect as lancedb_connect,
)
from openai import Client as OpenAIClient
from pydantic import Field, PrivateAttr

from crewai_tools.tools.rag.rag_tool import Adapter


def _default_embedding_function() -> Callable[[list[str]], list[list[float]]]:
    """Create a default embedding function using OpenAI."""
    client = OpenAIClient()

    def _embedding_function(input: list[str]) -> list[list[float]]:
        rs = client.embeddings.create(input=input, model="text-embedding-ada-002")
        return [record.embedding for record in rs.data]

    return _embedding_function


class LanceDBAdapter(Adapter):
    uri: str | Path
    table_name: str
    embedding_function: Callable[[list[str]], list[list[float]]] = Field(
        default_factory=_default_embedding_function
    )
    top_k: int = 3
    vector_column_name: str = "vector"
    text_column_name: str = "text"

    _db: Any = PrivateAttr()
    _table: Any = PrivateAttr()
    _lock_name: str = PrivateAttr(default="")

    def model_post_init(self, __context: Any) -> None:
        self._db = lancedb_connect(self.uri)
        self._table = self._db.open_table(self.table_name)
        self._lock_name = f"lancedb:{os.path.realpath(str(self.uri))}"

        super().model_post_init(__context)

    def query(self, question: str) -> str:  # type: ignore[override]
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
        with store_lock(self._lock_name):
            self._table.add(*args, **kwargs)

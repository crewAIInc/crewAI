from pathlib import Path
from typing import Any, Callable

from lancedb import DBConnection as LanceDBConnection
from lancedb import connect as lancedb_connect
from lancedb.table import Table as LanceDBTable
from openai import Client as OpenAIClient
from pydantic import Field, PrivateAttr

from crewai_tools.tools.rag.rag_tool import Adapter


def _default_embedding_function():
    client = OpenAIClient()

    def _embedding_function(input):
        rs = client.embeddings.create(input=input, model="text-embedding-ada-002")
        return [record.embedding for record in rs.data]

    return _embedding_function


class LanceDBAdapter(Adapter):
    uri: str | Path
    table_name: str
    embedding_function: Callable = Field(default_factory=_default_embedding_function)
    top_k: int = 3
    vector_column_name: str = "vector"
    text_column_name: str = "text"

    _db: LanceDBConnection = PrivateAttr()
    _table: LanceDBTable = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._db = lancedb_connect(self.uri)
        self._table = self._db.open_table(self.table_name)

        super().model_post_init(__context)

    def query(self, question: str) -> str:
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
        self._table.add(*args, **kwargs)

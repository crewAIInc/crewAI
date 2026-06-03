"""Oracle embedding function implementation."""

from __future__ import annotations

import json
from typing import Any

from crewai.rag.core.base_embeddings_callable import EmbeddingFunction
from crewai.rag.core.types import Documents, Embeddings


class OracleEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function backed by Oracle Database calls."""

    def __init__(
        self,
        *,
        conn: Any | None = None,
        connection_params: dict[str, Any] | None = None,
        embedding_params: dict[str, Any],
        proxy: str | None = None,
    ) -> None:
        try:
            import oracledb  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "oracledb is required for oracle embeddings. Install it with: uv add oracledb"
            ) from e

        self._oracledb = oracledb
        self._embedding_params = embedding_params
        self._proxy = proxy
        self._owns_connection = conn is None
        self._conn = conn or oracledb.connect(**(connection_params or {}))

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "oracle"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents using Oracle Database."""
        if isinstance(input, str):
            input = [input]
        if not input:
            raise ValueError("Oracle embeddings input cannot be empty.")

        cursor = None
        try:
            # Return strings/bytes for JSON payloads instead of locators.
            self._oracledb.defaults.fetch_lobs = False
            cursor = self._conn.cursor()

            if self._proxy:
                cursor.execute("begin utl_http.set_proxy(:proxy); end;", proxy=self._proxy)

            chunks = [
                json.dumps({"chunk_id": i, "chunk_data": text})
                for i, text in enumerate(input, start=1)
            ]
            vector_array_type = self._conn.gettype("SYS.VECTOR_ARRAY_T")
            inputs = vector_array_type.newobject(chunks)

            cursor.setinputsizes(None, self._oracledb.DB_TYPE_JSON)
            cursor.execute(
                "select t.* from dbms_vector_chain.utl_to_embeddings(:1, json(:2)) t",
                [inputs, self._embedding_params],
            )

            embeddings: list[list[float]] = []
            for row in cursor:
                if row is None:
                    raise ValueError("Oracle embeddings returned an empty row.")
                parsed = json.loads(row[0])
                embeddings.append(json.loads(parsed["embed_vector"]))
            return embeddings
        finally:
            if cursor is not None:
                cursor.close()

    def __del__(self) -> None:
        try:
            if getattr(self, "_owns_connection", False):
                conn = getattr(self, "_conn", None)
                if conn is not None:
                    conn.close()
        except Exception:
            pass

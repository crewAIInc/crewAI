"""Oracle embedding function implementation."""

from __future__ import annotations

from contextlib import suppress
import json
from typing import Any

import numpy as np

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
            import oracledb
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
        """Embed documents of at most 4000 characters, aligned by Oracle embed_id.

        Missing, duplicate, or unexpected result IDs raise ValueError. LOB
        payloads are read locally without changing process-wide driver defaults.
        """
        if isinstance(input, str):
            input = [input]
        if not input:
            raise ValueError("Oracle embeddings input cannot be empty.")

        if any(len(text) > 4000 for text in input):
            raise ValueError(
                "Oracle embedding documents must not exceed 4000 characters."
            )

        cursor = None
        try:
            cursor = self._conn.cursor()

            if self._proxy:
                cursor.execute(
                    "begin utl_http.set_proxy(:proxy); end;", proxy=self._proxy
                )

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

            embeddings_by_id: dict[int, list[float]] = {}
            for row in cursor:
                if row is None:
                    raise ValueError("Oracle embeddings returned an empty row.")
                payload = row[0]
                if hasattr(payload, "read"):
                    payload = payload.read()
                parsed = json.loads(payload)
                embed_id = parsed.get("embed_id")
                if (
                    type(embed_id) is not int
                    or not 1 <= embed_id <= len(input)
                    or embed_id in embeddings_by_id
                ):
                    raise ValueError(
                        "Oracle embeddings returned an invalid or duplicate embed_id."
                    )
                embeddings_by_id[embed_id] = json.loads(parsed["embed_vector"])
            if len(embeddings_by_id) != len(input):
                raise ValueError(
                    "Oracle embeddings count does not match the input documents."
                )
            return [
                np.asarray(embeddings_by_id[i], dtype=np.float32)
                for i in range(1, len(input) + 1)
            ]
        finally:
            if cursor is not None:
                if self._proxy and not self._owns_connection:
                    with suppress(Exception):
                        cursor.execute("begin utl_http.set_proxy(NULL); end;")
                cursor.close()

    def __del__(self) -> None:
        # Destructors must tolerate partially initialized or closed connections.
        with suppress(Exception):
            if getattr(self, "_owns_connection", False):
                conn = getattr(self, "_conn", None)
                if conn is not None:
                    conn.close()

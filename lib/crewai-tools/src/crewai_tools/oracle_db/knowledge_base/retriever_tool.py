from __future__ import annotations

import array
import json
import re
from typing import Any, Literal

from crewai.rag.embeddings.factory import build_embedder
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from crewai_tools.oracle_db.common import (
    get_oracle_connection_kwargs,
    oracle_connection_context,
    validate_identifier,
)


def _generate_accum_query(query: str, fuzzy: bool = False) -> str:
    words = re.split(r"\W+", query)
    tokens = [word for word in words if word]
    if fuzzy:
        return " ACCUM ".join(f'fuzzy("{token}")' for token in tokens)
    return " ACCUM ".join(f'"{token}"' for token in tokens)


class OracleSearchToolInput(BaseModel):
    query: str = Field(..., description="The query to retrieve information from Oracle.")


class OracleToolBase(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    client: Any | None = Field(default=None, exclude=True)
    user: str | None = Field(default=None, description="Oracle DB username")
    password: str | None = Field(default=None, description="Oracle DB password")
    dsn: str | None = Field(default=None, description="Oracle DB DSN")
    config_dir: str | None = Field(default=None, description="Optional Oracle config dir")
    wallet_location: str | None = Field(
        default=None, description="Optional wallet directory"
    )
    wallet_password: str | None = Field(
        default=None, description="Optional wallet password"
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["oracledb"])

    def _connection_kwargs(self) -> dict[str, Any]:
        if self.client is not None:
            return {}
        return get_oracle_connection_kwargs(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            config_dir=self.config_dir,
            wallet_location=self.wallet_location,
            wallet_password=self.wallet_password,
        )

    def _result_json(self, results: list[dict[str, Any]]) -> str:
        if results:
            return json.dumps({"results": results}, indent=2)
        return json.dumps({"message": "No results found for the given query."}, indent=2)


class OracleVectorSearchTool(OracleToolBase):
    name: str = "Oracle Vector Search Tool"
    description: str = (
        "Retrieves information from Oracle Database vector columns using "
        "VECTOR_DISTANCE against an externally generated embedding."
    )
    args_schema: type[BaseModel] = OracleSearchToolInput
    table_name: str = Field(..., description="Oracle table that stores the documents")
    text_column: str = Field(default="text", description="Text column to return")
    embedding_column: str = Field(
        default="embedding", description="Vector column to compare against"
    )
    metadata_column: str | None = Field(
        default="metadata",
        description="Optional JSON metadata column to merge into each result",
    )
    metadata_columns: list[str] = Field(
        default_factory=list,
        description="Additional scalar columns to return as metadata",
    )
    number_of_results: int = Field(default=5, description="Maximum results to return")
    distance_metric: Literal["COSINE", "EUCLIDEAN", "DOT"] = Field(
        default="COSINE", description="Oracle VECTOR_DISTANCE metric"
    )
    embedding_model: dict[str, Any] | None = Field(
        default=None,
        exclude=True,
        description="Optional CrewAI embedder specification used to build query embeddings",
    )
    embedder: Any | None = Field(
        default=None,
        exclude=True,
        description="Optional prebuilt embedding callable",
    )

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.embedder is None and self.embedding_model is not None:
            self.embedder = build_embedder(self.embedding_model)

    def _embed_query(self, query: str) -> array.array[float]:
        if self.embedder is None:
            raise ValueError(
                "OracleVectorSearchTool requires either embedder or embedding_model."
            )

        embedding_response = self.embedder([query])
        if not embedding_response:
            raise ValueError("Embedding model returned no vectors.")

        embedding = embedding_response[0]
        return array.array("f", [float(value) for value in embedding])

    def _run(self, query: str) -> str:
        table_name = validate_identifier(self.table_name, field_name="table_name")
        text_column = validate_identifier(self.text_column, field_name="text_column")
        embedding_column = validate_identifier(
            self.embedding_column, field_name="embedding_column"
        )
        metadata_column = None
        if self.metadata_column:
            metadata_column = validate_identifier(
                self.metadata_column, field_name="metadata_column"
            )
        metadata_columns = [
            validate_identifier(column, field_name="metadata_columns")
            for column in self.metadata_columns
            if column.lower() != text_column.lower()
            and column.lower() != embedding_column.lower()
            and (metadata_column is None or column.lower() != metadata_column.lower())
        ]

        fetch_columns = [text_column]
        if metadata_column:
            fetch_columns.append(metadata_column)
        fetch_columns.extend(metadata_columns)
        fetch_columns_sql = ", ".join(fetch_columns)
        number_of_results = max(1, self.number_of_results)
        metric = self.distance_metric.upper()
        sql = (
            f"SELECT {fetch_columns_sql}, "  # noqa: S608
            f"VECTOR_DISTANCE({embedding_column}, :query_embedding, {metric}) distance "
            f"FROM {table_name} ORDER BY distance ASC "
            f"FETCH FIRST {number_of_results} ROWS ONLY"
        )

        with oracle_connection_context(self.client, **self._connection_kwargs()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, query_embedding=self._embed_query(query))
                columns = [column[0].lower() for column in cursor.description]
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row, strict=False))
                    metadata: dict[str, Any] = {}
                    if metadata_column:
                        metadata_value = row_dict.get(metadata_column.lower())
                        if isinstance(metadata_value, dict):
                            metadata.update(metadata_value)
                        elif isinstance(metadata_value, str):
                            try:
                                parsed_metadata = json.loads(metadata_value)
                            except json.JSONDecodeError:
                                parsed_metadata = None
                            if isinstance(parsed_metadata, dict):
                                metadata.update(parsed_metadata)
                    metadata.update(
                        {
                            column: row_dict.get(column.lower())
                            for column in metadata_columns
                        }
                    )
                    results.append(
                        {
                            "content": row_dict[text_column.lower()],
                            "metadata": metadata,
                            "distance": row_dict.get("distance"),
                        }
                    )

        return self._result_json(results)


class OracleTextSearchTool(OracleToolBase):
    name: str = "Oracle Text Search Tool"
    description: str = (
        "Retrieves information from Oracle Database using Oracle Text CONTAINS search."
    )
    args_schema: type[BaseModel] = OracleSearchToolInput
    table_name: str = Field(..., description="Oracle table that stores the documents")
    text_column: str = Field(default="text", description="Text column to search")
    metadata_columns: list[str] = Field(
        default_factory=list,
        description="Additional columns to return as metadata alongside the text",
    )
    number_of_results: int = Field(default=5, description="Maximum results to return")
    operator_search: bool = Field(
        default=False,
        description="Treat the query as a raw Oracle Text expression instead of ACCUM tokens",
    )
    fuzzy: bool = Field(
        default=False,
        description="Apply Oracle Text FUZZY matching when operator_search is false",
    )
    return_scores: bool = Field(
        default=True, description="Include Oracle Text SCORE(1) in each result"
    )

    def _run(self, query: str) -> str:
        table_name = validate_identifier(self.table_name, field_name="table_name")
        text_column = validate_identifier(self.text_column, field_name="text_column")
        metadata_columns = [
            validate_identifier(column, field_name="metadata_columns")
            for column in self.metadata_columns
            if column.lower() != text_column.lower()
        ]
        number_of_results = max(1, self.number_of_results)

        search_text = query if self.operator_search else _generate_accum_query(query, self.fuzzy)
        if not search_text:
            return self._result_json([])

        select_columns = [text_column, *metadata_columns]
        select_columns_sql = ", ".join(select_columns)
        sql = (
            f"SELECT SCORE(1) score, {select_columns_sql} FROM {table_name} "  # noqa: S608
            f"WHERE CONTAINS({text_column}, :query, 1) > 0 "
            f"ORDER BY score DESC FETCH FIRST {number_of_results} ROWS ONLY"
        )

        with oracle_connection_context(self.client, **self._connection_kwargs()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, query=search_text)
                columns = [column[0].lower() for column in cursor.description]
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(zip(columns, row, strict=False))
                    result = {
                        "content": row_dict[text_column.lower()],
                        "metadata": {
                            column: row_dict.get(column.lower()) for column in metadata_columns
                        },
                    }
                    if self.return_scores:
                        result["score"] = row_dict.get("score")
                    results.append(result)

        return self._result_json(results)


class OracleHybridSearchTool(OracleToolBase):
    name: str = "Oracle Hybrid Search Tool"
    description: str = (
        "Retrieves information from Oracle Database hybrid vector indexes using "
        "DBMS_HYBRID_VECTOR.SEARCH."
    )
    args_schema: type[BaseModel] = OracleSearchToolInput
    hybrid_index_name: str = Field(..., description="Hybrid index name to query")
    table_name: str = Field(..., description="Oracle table that stores the documents")
    text_column: str = Field(default="text", description="Text column to return")
    metadata_column: str | None = Field(
        default="metadata",
        description="Optional JSON metadata column to merge into each result",
    )
    metadata_columns: list[str] = Field(
        default_factory=list,
        description="Additional scalar columns to return as metadata",
    )
    number_of_results: int = Field(default=5, description="Maximum results to return")
    search_mode: Literal["keyword", "hybrid", "semantic"] = Field(
        default="hybrid", description="Oracle hybrid search mode"
    )
    return_scores: bool = Field(
        default=True, description="Include hybrid, vector, and text scores"
    )
    params: dict[str, Any] | None = Field(
        default=None, description="Additional DBMS_HYBRID_VECTOR.SEARCH parameters"
    )

    def _build_search_params(self, query: str) -> dict[str, Any]:
        search_params = dict(self.params or {})
        search_params["hybrid_index_name"] = validate_identifier(
            self.hybrid_index_name, field_name="hybrid_index_name"
        )

        if "return" in search_params or "search_text" in search_params:
            raise ValueError("Reserved hybrid search params cannot be supplied directly.")

        if self.search_mode in {"hybrid", "semantic"}:
            search_params["vector"] = dict(search_params.get("vector") or {})
            if "search_text" in search_params["vector"] or "search_vector" in search_params["vector"]:
                raise ValueError("vector.search_text and vector.search_vector are managed internally.")
            search_params["vector"]["search_text"] = query

        if self.search_mode in {"hybrid", "keyword"}:
            search_params["text"] = dict(search_params.get("text") or {})
            if (
                "search_text" in search_params["text"]
                or "search_vector" in search_params["text"]
                or "contains" in search_params["text"]
            ):
                raise ValueError("text.search_text, text.search_vector, and text.contains are managed internally.")
            search_params["text"]["search_text"] = query

        search_params["return"] = {
            "topN": max(1, self.number_of_results),
            "values": ["rowid", "score", "vector_score", "text_score"],
            "format": "JSON",
        }
        return search_params

    def _run(self, query: str) -> str:
        table_name = validate_identifier(self.table_name, field_name="table_name")
        text_column = validate_identifier(self.text_column, field_name="text_column")
        metadata_column = None
        if self.metadata_column:
            metadata_column = validate_identifier(
                self.metadata_column, field_name="metadata_column"
            )
        metadata_columns = [
            validate_identifier(column, field_name="metadata_columns")
            for column in self.metadata_columns
            if column.lower() != text_column.lower()
            and (metadata_column is None or column.lower() != metadata_column.lower())
        ]
        search_params = self._build_search_params(query)

        fetch_columns = [text_column]
        if metadata_column:
            fetch_columns.append(metadata_column)
        fetch_columns.extend(metadata_columns)
        fetch_columns_sql = ", ".join(fetch_columns)
        row_sql = f"SELECT {fetch_columns_sql} FROM {table_name} WHERE rowid = :1"  # noqa: S608

        with oracle_connection_context(self.client, **self._connection_kwargs()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT DBMS_HYBRID_VECTOR.SEARCH(json(:search_params))",
                    search_params=json.dumps(search_params),
                )
                raw = cursor.fetchall()
                if not raw:
                    return self._result_json([])
                raw_payload = raw[0][0]
                if hasattr(raw_payload, "read"):
                    raw_payload = raw_payload.read()
                rowids = json.loads(raw_payload)

                results = []
                for item in rowids:
                    cursor.execute(row_sql, [item["rowid"]])
                    row = cursor.fetchone()
                    if row is None:
                        continue

                    row_index = 0
                    content = row[row_index]
                    row_index += 1
                    metadata: dict[str, Any] = {}
                    if metadata_column:
                        metadata_value = row[row_index]
                        row_index += 1
                        if isinstance(metadata_value, dict):
                            metadata.update(metadata_value)
                        elif metadata_value is not None:
                            metadata[metadata_column] = metadata_value
                    for column in metadata_columns:
                        metadata[column] = row[row_index]
                        row_index += 1

                    result = {
                        "content": content,
                        "metadata": metadata,
                    }
                    if self.return_scores:
                        result["score"] = item.get("score")
                        result["vector_score"] = item.get("vector_score")
                        result["text_score"] = item.get("text_score")
                    results.append(result)

        return self._result_json(results)

from __future__ import annotations

import array
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from decimal import Decimal
import json
import logging
import os
import re
from typing import Any, Literal
import uuid

from crewai.tools import BaseTool, EnvVar
from openai import AzureOpenAI, Client
from pydantic import BaseModel, ConfigDict, Field, model_validator


try:
    import oracledb

    ORACLEDB_AVAILABLE = True
except ImportError:
    ORACLEDB_AVAILABLE = False
    oracledb = Any  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r'^(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$#]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_$#]*))*$')
_JSON_PATH_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
_ID_COLUMN = "id"
_TEXT_COLUMN = "text"
_METADATA_COLUMN = "metadata"
_EMBEDDING_COLUMN = "embedding"
LOGICAL_MAP = {
    "$and": (" AND ", "({0})"),
    "$or": (" OR ", "({0})"),
    "$nor": (" OR ", "( NOT ({0}) )"),
}
COMPARISON_MAP = {
    "$exists": "",
    "$eq": "@ == {0}",
    "$ne": "@ != {0}",
    "$gt": "@ > {0}",
    "$lt": "@ < {0}",
    "$gte": "@ >= {0}",
    "$lte": "@ <= {0}",
    "$between": "",
    "$startsWith": "@ starts with {0}",
    "$hasSubstring": "@ has substring {0}",
    "$instr": "@ has substring {0}",
    "$regex": "@ like_regex {0}",
    "$like": "@ like {0}",
    "$in": "",
    "$nin": "",
    "$all": "",
    "$not": "",
}
NOT_OPERS = ["$nin", "$not", "$exists"]


def _read_lob_if_needed(value: Any) -> Any:
    """Read Oracle LOB values eagerly so downstream JSON/text handling works."""
    if hasattr(value, "read") and callable(value.read):
        return value.read()
    return value


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return value


def _quote_identifier(name: str) -> str:
    name = name.strip()
    if not _IDENTIFIER_RE.fullmatch(name):
        raise ValueError(f"Identifier '{name}' is not valid for Oracle SQL.")

    parts: list[str] = []
    current = []
    in_quotes = False
    for char in name:
        if char == '"':
            in_quotes = not in_quotes
            current.append(char)
            continue
        if char == "." and not in_quotes:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))

    quoted_parts = []
    for part in parts:
        stripped = part.strip()
        if stripped.startswith('"') and stripped.endswith('"'):
            quoted_parts.append(stripped)
        else:
            quoted_parts.append(f'"{stripped}"')
    return ".".join(quoted_parts)


def _validate_metadata_key(metadata_key: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_\.\[\],\s\*]*", metadata_key):
        raise ValueError(
            f"Invalid metadata key '{metadata_key}'. Only letters, numbers, underscores, nesting via '.', and array wildcards '[*]' are allowed."
        )


def _validate_int_param(
    config: dict[str, Any],
    key: str,
    min_value: int,
    max_value: int | None = None,
) -> None:
    if key not in config:
        return

    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    if value < min_value:
        raise ValueError(f"{key} must be at least {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{key} must be at most {max_value}.")


def _validate_params(
    config: dict[str, Any],
    allowed_keys: set[str],
) -> None:
    for key in config:
        if key not in allowed_keys:
            raise ValueError(f"Invalid parameter: {key}")


@contextmanager
def _get_connection(client: Any) -> Any:
    connection_pool_class = getattr(oracledb, "ConnectionPool", None)
    if connection_pool_class is not None and isinstance(client, connection_pool_class):
        with client.acquire() as connection:
            yield connection
        return

    if hasattr(client, "cursor") and callable(client.cursor):
        yield client
        return

    if hasattr(client, "acquire") and callable(client.acquire):
        with client.acquire() as connection:
            yield connection
        return

    raise TypeError(
        "client must be an oracledb connection or connection pool compatible object."
    )


def _get_comparison_string(
    oper: str, value: Any, bind_variables: list[Any]
) -> tuple[str, str]:
    if oper not in COMPARISON_MAP:
        raise ValueError(f"Invalid operator: {oper}")

    if COMPARISON_MAP[oper] != "":
        bind_index = len(bind_variables)
        bind_variables.append(value)
        return (
            COMPARISON_MAP[oper].format(f"$val{bind_index}"),
            f':value{bind_index} AS "val{bind_index}"',
        )

    if oper == "$between":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(
                f"Invalid value for $between: {value}. It must be a list containing exactly 2 elements."
            )

        min_val, max_val = value
        if min_val is None and max_val is None:
            raise ValueError("At least one bound in $between must be non-null.")

        conditions: list[str] = []
        passings: list[str] = []

        if min_val is not None:
            bind_index = len(bind_variables)
            bind_variables.append(min_val)
            conditions.append(f"@ >= $val{bind_index}")
            passings.append(f':value{bind_index} AS "val{bind_index}"')

        if max_val is not None:
            bind_index = len(bind_variables)
            bind_variables.append(max_val)
            conditions.append(f"@ <= $val{bind_index}")
            passings.append(f':value{bind_index} AS "val{bind_index}"')

        return " && ".join(conditions), ",".join(passings)

    if oper in ["$in", "$nin", "$all"]:
        if not isinstance(value, list):
            raise ValueError(
                f"Invalid value for {oper}: {value}. It must be a non-empty list."
            )

        value_binds: list[str] = []
        passings: list[str] = []
        for current_value in value:
            bind_index = len(bind_variables)
            bind_variables.append(current_value)
            value_binds.append(f"$val{bind_index}")
            passings.append(f':value{bind_index} AS "val{bind_index}"')

        if oper == "$all":
            condition = "@ == " + " && @ == ".join(value_binds)
        else:
            condition = f"@ in ({','.join(value_binds)})"

        return condition, ",".join(passings)

    raise ValueError(f"Invalid operator: {oper}.")


def _generate_condition(metadata_key: str, value: Any, bind_variables: list[Any]) -> str:
    single_mask = (
        "JSON_EXISTS(metadata, '$.{key}?(@ {oper} $val)' "
        'PASSING {value_bind} AS "val")'
    )
    multiple_mask = "JSON_EXISTS(metadata, '$.{key}?({filters})' PASSING {passes})"

    _validate_metadata_key(metadata_key)

    if not isinstance(value, (dict, list, tuple)):
        bind_name = f":value{len(bind_variables)}"
        bind_variables.append(value)
        return single_mask.format(key=metadata_key, oper="==", value_bind=bind_name)

    if isinstance(value, dict):
        if not all(value_key.startswith("$") for value_key in value):
            raise ValueError("Nested metadata objects are not supported in filters.")

        not_dict: dict[str, Any] = {}
        passing_values: list[str] = []
        comparison_values: list[str] = []
        all_conditions: list[str] = []

        for oper, current_value in value.items():
            if (
                oper in NOT_OPERS
                or (oper == "$eq" and isinstance(current_value, (list, dict)))
                or (oper == "$ne" and isinstance(current_value, (list, dict)))
            ):
                not_dict[oper] = current_value
                continue

            result, passings = _get_comparison_string(
                oper, current_value, bind_variables
            )
            comparison_values.append(result)
            passing_values.append(passings)

        if comparison_values:
            all_conditions.append(
                multiple_mask.format(
                    key=metadata_key,
                    filters=" && ".join(comparison_values),
                    passes=" , ".join(passing_values),
                )
            )

        for oper, current_value in not_dict.items():
            if oper == "$not":
                all_conditions.append(
                    f"NOT ({_generate_condition(metadata_key, current_value, bind_variables)})"
                )
            elif oper == "$exists":
                if not isinstance(current_value, bool):
                    raise ValueError(
                        f"Invalid value for $exists: {current_value}. It must be a boolean."
                    )
                if current_value:
                    all_conditions.append(f"JSON_EXISTS(metadata, '$.{metadata_key}')")
                else:
                    all_conditions.append(
                        f"NOT (JSON_EXISTS(metadata, '$.{metadata_key}'))"
                    )
            elif oper == "$nin":
                result, passings = _get_comparison_string(
                    oper, current_value, bind_variables
                )
                all_conditions.append(
                    " NOT "
                    + multiple_mask.format(
                        key=metadata_key, filters=result, passes=passings
                    )
                )
            elif oper == "$eq":
                bind_index = len(bind_variables)
                bind_variables.append(json.dumps(current_value))
                all_conditions.append(
                    f"JSON_EQUAL(JSON_QUERY(metadata, '$.{metadata_key}'), JSON(:value{bind_index}))"
                )
            elif oper == "$ne":
                bind_index = len(bind_variables)
                bind_variables.append(json.dumps(current_value))
                all_conditions.append(
                    f"NOT (JSON_EQUAL(JSON_QUERY(metadata, '$.{metadata_key}'), JSON(:value{bind_index})))"
                )

        result = " AND ".join(all_conditions)
        if len(all_conditions) > 1:
            return f"({result})"
        return result

    raise ValueError("Filter format is invalid.")


def _generate_where_clause(filter_spec: dict[str, Any], bind_variables: list[Any]) -> str:
    if not isinstance(filter_spec, dict):
        raise ValueError("Filter syntax is incorrect. Must be a dictionary.")

    all_conditions: list[str] = []
    for key, value in filter_spec.items():
        if key.startswith("$"):
            if key not in LOGICAL_MAP:
                raise ValueError(f"'{key}' is not a recognized logical operator.")
            if not isinstance(value, list):
                raise ValueError("Logical operators require an array of values.")
            joiner, wrapper = LOGICAL_MAP[key]
            combine_conditions = [
                _generate_where_clause(item, bind_variables) for item in value
            ]
            all_conditions.append(wrapper.format(joiner.join(combine_conditions)))
        else:
            all_conditions.append(_generate_condition(key, value, bind_variables))

    result = " AND ".join(all_conditions)
    if len(all_conditions) > 1:
        return f"({result})"
    return result


def _build_metadata_filter(
    filter_by: str | None,
    filter_value: Any | None,
    filter_spec: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    bind_variables: list[Any] = []

    if filter_by is None and filter_value is None:
        simple_params: dict[str, Any] = {}
    else:
        if filter_by is None or filter_value is None:
            raise ValueError("filter_by and filter_value must be provided together.")
        if not _JSON_PATH_RE.fullmatch(filter_by):
            raise ValueError(
                "filter_by must reference a metadata path using letters, numbers, underscores, and dots only."
            )
        clauses.append(_generate_condition(filter_by, filter_value, bind_variables))
        simple_params = {}

    if filter_spec:
        clauses.append(_generate_where_clause(filter_spec, bind_variables))

    params = dict(simple_params)
    for i, value in enumerate(bind_variables):
        params[f"value{i}"] = value

    return " AND ".join(clauses), params


def _extract_oracle_error_code(exc: Exception) -> int | None:
    error = getattr(exc, "args", [None])[0]
    return getattr(error, "code", None)


class OracleToolSchema(BaseModel):
    query: str = Field(
        ..., description="Query to search in the Oracle vector store - always required."
    )
    filter_by: str | None = Field(
        default=None,
        description="Metadata field path to filter by, for example 'source' or 'department.region'.",
    )
    filter_value: Any | None = Field(
        default=None,
        description="Metadata value to filter by. Must be used together with filter_by.",
    )
    filters: str | None = Field(
        default=None,
        description="Optional JSON string with Oracle-style metadata filters using operators like $and, $or, $eq, $gt, $in, or $exists.",
    )
    limit: int | None = Field(
        default=None,
        description="Optional result limit overriding the tool default for this search.",
    )
    score_threshold: float | None = Field(
        default=None,
        description="Optional maximum vector distance. Only rows with distance less than or equal to this value are returned.",
    )


class OracleVectorSearchQueryConfig(BaseModel):
    """Default query behavior for Oracle vector search."""

    limit: int | None = Field(default=None, ge=1)
    score_threshold: float | None = Field(default=None, ge=0.0)
    filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional Oracle-style metadata filter dictionary.",
    )


class OracleVectorSearchConfig(BaseModel):
    """Oracle connection and vector search settings."""

    user: str | None = None
    password: str | None = None
    dsn: str | None = None
    table_name: str
    limit: int = 3
    score_threshold: float | None = Field(default=None, ge=0.0)
    distance_strategy: Literal["COSINE", "EUCLIDEAN", "DOT"] = "COSINE"
    index_name: str | None = None
    connection_kwargs: dict[str, Any] = Field(default_factory=dict)


class OracleVectorSearchTool(BaseTool):
    """Search Oracle AI Vector Search tables for relevant documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "OracleVectorSearchTool"
    description: str = (
        "Search Oracle AI Vector Search tables for semantically relevant documents."
    )
    args_schema: type[BaseModel] = OracleToolSchema
    package_dependencies: list[str] = Field(default_factory=lambda: ["oracledb"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENAI_API_KEY",
                description="API key for default OpenAI embeddings when embedding_function is not provided",
                required=True,
            )
        ]
    )
    oracle_config: OracleVectorSearchConfig
    query_config: OracleVectorSearchQueryConfig | None = Field(
        default=None,
        description="Optional default query behavior including filters and maximum distance threshold.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model used when embedding_function is not provided.",
    )
    dimensions: int = Field(
        default=3072,
        description="Embedding dimension used when creating Oracle vector tables and indexes.",
    )
    embedding_function: Callable[[str], list[float]] | None = Field(
        default=None,
        description="Optional embedding function used instead of OpenAI embeddings.",
    )
    client: Any | None = Field(default=None, description="Optional pre-configured Oracle connection.")
    _openai_client: Any | None = None
    _owns_client: bool = False

    @model_validator(mode="after")
    def _setup_oracle(self) -> OracleVectorSearchTool:
        global ORACLEDB_AVAILABLE, oracledb

        if not ORACLEDB_AVAILABLE:
            import click

            if click.confirm(
                "The 'oracledb' package is required to use OracleVectorSearchTool. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "oracledb"], check=True)  # noqa: S607
                try:
                    import oracledb as installed_oracledb
                except ImportError as exc:
                    raise ImportError(
                        "The 'oracledb' package is required to use OracleVectorSearchTool. Please install it with: uv add oracledb"
                    ) from exc

                oracledb = installed_oracledb
                ORACLEDB_AVAILABLE = True
            else:
                raise ImportError(
                    "The 'oracledb' package is required to use OracleVectorSearchTool. Please install it with: uv add oracledb"
                )

        if self.client is None:
            connect_params = dict(self.oracle_config.connection_kwargs)
            if self.oracle_config.user is not None:
                connect_params["user"] = self.oracle_config.user
            if self.oracle_config.password is not None:
                connect_params["password"] = self.oracle_config.password
            if self.oracle_config.dsn is not None:
                connect_params["dsn"] = self.oracle_config.dsn
            self.client = oracledb.connect(**connect_params)
            self._owns_client = True

        if self.embedding_function is None:
            if "AZURE_OPENAI_ENDPOINT" in os.environ:
                self._openai_client = AzureOpenAI()
            elif "OPENAI_API_KEY" in os.environ:
                self._openai_client = Client()
            else:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required for OracleVectorSearchTool unless embedding_function is provided."
                )

        return self

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.embedding_function is not None:
            return [self.embedding_function(text) for text in texts]
        return [
            item.embedding
            for item in self._openai_client.embeddings.create(
                input=texts,
                model=self.embedding_model,
                dimensions=self.dimensions,
            ).data
        ]

    def _table_exists(self, table_name: str) -> bool:
        try:
            with _get_connection(self.client) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT 1 FROM {table_name} WHERE ROWNUM < 1")
                    return True
        except Exception as exc:
            if _extract_oracle_error_code(exc) == 942:
                return False
            raise

    def _index_exists(self, index_name: str, table_name: str | None = None) -> bool:
        index_name_no_quotes = index_name.replace('"', "")
        table_name_no_quotes = table_name.replace('"', "") if table_name else None
        query = """
            SELECT index_name
            FROM all_indexes
            WHERE index_name = :idx_name
        """
        if table_name_no_quotes:
            query += " AND table_name = :table_name"

        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                if table_name_no_quotes:
                    cursor.execute(
                        query,
                        idx_name=index_name_no_quotes,
                        table_name=table_name_no_quotes,
                    )
                else:
                    cursor.execute(query, idx_name=index_name_no_quotes)
                return cursor.fetchone() is not None

    def table_exists(self) -> bool:
        return self._table_exists(_quote_identifier(self.oracle_config.table_name))

    def vector_index_exists(self, index_name: str | None = None) -> bool:
        effective_index_name = index_name or self.oracle_config.index_name
        if effective_index_name is None:
            base_name = self.oracle_config.table_name.split(".")[-1].replace('"', "")
            effective_index_name = f"{base_name}_HNSW_IDX"
        return self._index_exists(
            _quote_identifier(effective_index_name),
            _quote_identifier(self.oracle_config.table_name).split(".")[-1],
        )

    def create_table(self) -> None:
        table_name = _quote_identifier(self.oracle_config.table_name)
        if self._table_exists(table_name):
            return

        ddl = (
            f"CREATE TABLE {table_name} ("
            f"{_ID_COLUMN} VARCHAR2(64) PRIMARY KEY, "
            f"{_TEXT_COLUMN} CLOB, "
            f"{_METADATA_COLUMN} JSON, "
            f"{_EMBEDDING_COLUMN} VECTOR({self.dimensions}, FLOAT32)"
            ")"
        )
        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
            connection.commit()

    def create_vector_index(
        self,
        *,
        index_name: str | None = None,
        idx_type: Literal["HNSW", "IVF"] = "HNSW",
        params: dict[str, Any] | None = None,
    ) -> str:
        table_name = _quote_identifier(self.oracle_config.table_name)
        if not isinstance(idx_type, str):
            raise ValueError("idx_type must be HNSW or IVF.")
        normalized_idx_type = idx_type.upper()
        if normalized_idx_type not in {"HNSW", "IVF"}:
            raise ValueError("idx_type must be HNSW or IVF.")

        effective_index_name = index_name or self.oracle_config.index_name
        if effective_index_name is None:
            base_name = self.oracle_config.table_name.split(".")[-1].replace('"', "")
            effective_index_name = f"{base_name}_{normalized_idx_type}_IDX"
        quoted_index_name = _quote_identifier(effective_index_name)

        raw_params = {} if params is None else params
        if not isinstance(raw_params, dict):
            raise ValueError("params must be a dictionary.")

        if normalized_idx_type == "HNSW":
            index_config = {
                "accuracy": 90,
                "neighbors": 32,
                "efconstruction": 200,
                "parallel": 8,
                **raw_params,
            }
            _validate_params(
                index_config,
                {"accuracy", "neighbors", "efconstruction", "parallel"},
            )
            _validate_int_param(index_config, "accuracy", 1, 100)
            _validate_int_param(index_config, "neighbors", 2, 2048)
            _validate_int_param(index_config, "efconstruction", 1, 65535)
            _validate_int_param(index_config, "parallel", 1)

            ddl = (
                f"CREATE VECTOR INDEX {quoted_index_name} "
                f"ON {table_name}({_EMBEDDING_COLUMN}) "
                "ORGANIZATION INMEMORY NEIGHBOR GRAPH "
                f"DISTANCE {self.oracle_config.distance_strategy} "
                f"WITH TARGET ACCURACY {index_config['accuracy']} "
                f"PARAMETERS (type HNSW, neighbors {index_config['neighbors']}, "
                f"efconstruction {index_config['efconstruction']}) "
                f"PARALLEL {index_config['parallel']}"
            )
        else:
            index_config = {
                "accuracy": 90,
                "neighbor_partitions": 32,
                "parallel": 8,
                **raw_params,
            }
            _validate_params(
                index_config,
                {
                    "accuracy",
                    "neighbor_partitions",
                    "samples_per_partition",
                    "min_vectors_per_partition",
                    "parallel",
                },
            )
            _validate_int_param(index_config, "accuracy", 1, 100)
            _validate_int_param(index_config, "neighbor_partitions", 1, 10000000)
            _validate_int_param(index_config, "samples_per_partition", 1)
            _validate_int_param(index_config, "min_vectors_per_partition", 0)
            _validate_int_param(index_config, "parallel", 1)

            parameters_clause = (
                "PARAMETERS (type IVF, "
                f"neighbor partitions {index_config['neighbor_partitions']}"
            )
            if "samples_per_partition" in index_config:
                parameters_clause += (
                    f", samples_per_partition {index_config['samples_per_partition']}"
                )
            if "min_vectors_per_partition" in index_config:
                parameters_clause += (
                    f", min_vectors_per_partition {index_config['min_vectors_per_partition']}"
                )
            parameters_clause += ")"

            ddl = (
                f"CREATE VECTOR INDEX {quoted_index_name} "
                f"ON {table_name}({_EMBEDDING_COLUMN}) "
                "ORGANIZATION NEIGHBOR PARTITIONS "
                f"DISTANCE {self.oracle_config.distance_strategy} "
                f"WITH TARGET ACCURACY {index_config['accuracy']} "
                f"{parameters_clause} "
                f"PARALLEL {index_config['parallel']}"
            )

        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.execute(ddl)
            connection.commit()
        return effective_index_name

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        text_list = list(texts)
        if not text_list:
            return []

        metadata_list = metadatas or [{} for _ in text_list]
        if len(metadata_list) != len(text_list):
            raise ValueError("metadatas must match the number of texts.")

        id_list = ids or [uuid.uuid4().hex for _ in text_list]
        if len(id_list) != len(text_list):
            raise ValueError("ids must match the number of texts.")

        self.create_table()
        embeddings = self._embed_texts(text_list)
        if any(len(embedding) != self.dimensions for embedding in embeddings):
            raise ValueError(
                "Embedding dimension mismatch. Update dimensions or provide embeddings with consistent length."
            )

        insert_sql = (
            f"INSERT INTO {_quote_identifier(self.oracle_config.table_name)} "
            f"({_ID_COLUMN}, {_TEXT_COLUMN}, {_METADATA_COLUMN}, {_EMBEDDING_COLUMN}) "
            "VALUES (:1, :2, :3, :4)"
        )

        with _get_connection(self.client) as connection:
            with connection.cursor() as cursor:
                cursor.setinputsizes(
                    None,
                    None,
                    oracledb.DB_TYPE_JSON,
                    oracledb.DB_TYPE_VECTOR,
                )
                for doc_id, text, metadata, embedding in zip(
                    id_list, text_list, metadata_list, embeddings, strict=False
                ):
                    cursor.execute(
                        insert_sql,
                        [
                            doc_id,
                            text,
                            metadata,
                            array.array("f", embedding),
                        ],
                    )
            connection.commit()
        return id_list

    def _run(
        self,
        query: str,
        filter_by: str | None = None,
        filter_value: Any | None = None,
        filters: str | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
    ) -> str:
        try:
            table_name = _quote_identifier(self.oracle_config.table_name)
            if not self.table_exists():
                raise ValueError(
                    f"Table '{self.oracle_config.table_name}' does not exist. Create it first or point the tool at an existing Oracle vector table."
                )

            filter_spec: dict[str, Any] | None = None
            if self.query_config and self.query_config.filter:
                filter_spec = self.query_config.filter.copy()
            if filters:
                parsed_filters = json.loads(filters)
                if not isinstance(parsed_filters, dict):
                    raise ValueError("filters must decode to a JSON object.")
                filter_spec = (
                    {"$and": [filter_spec, parsed_filters]}
                    if filter_spec is not None
                    else parsed_filters
                )

            where_clause, where_params = _build_metadata_filter(
                filter_by,
                filter_value,
                filter_spec,
            )

            effective_limit = (
                limit
                if limit is not None
                else self.query_config.limit
                if self.query_config and self.query_config.limit is not None
                else self.oracle_config.limit
            )
            effective_score_threshold = (
                score_threshold
                if score_threshold is not None
                else self.query_config.score_threshold
                if self.query_config and self.query_config.score_threshold is not None
                else self.oracle_config.score_threshold
            )

            query_vector = array.array("f", self._embed_texts([query])[0])
            search_sql = f"""
                SELECT
                    {_TEXT_COLUMN},
                    {_METADATA_COLUMN},
                    vector_distance(
                        {_EMBEDDING_COLUMN},
                        :embedding,
                        {self.oracle_config.distance_strategy}
                    ) AS distance
                FROM {table_name}
                {"WHERE " + where_clause if where_clause else ""}
                ORDER BY distance
                FETCH APPROX FIRST {effective_limit} ROWS ONLY
            """

            with _get_connection(self.client) as connection:
                with connection.cursor() as cursor:
                    params = {"embedding": query_vector, **where_params}
                    cursor.execute(search_sql, params)
                    rows = cursor.fetchall()

            formatted_rows = []
            for text, metadata, distance in rows:
                text = _read_lob_if_needed(text)
                metadata = _read_lob_if_needed(metadata)
                distance_value = float(distance)

                if (
                    effective_score_threshold is not None
                    and distance_value > effective_score_threshold
                ):
                    continue

                metadata_dict: dict[str, Any]
                if isinstance(metadata, str) and metadata:
                    metadata_dict = _normalize_json_value(json.loads(metadata))
                else:
                    metadata_dict = _normalize_json_value(metadata or {})

                formatted_rows.append(
                    {
                        "distance": distance_value,
                        "score": distance_value,
                        "metadata": metadata_dict,
                        "context": text or "",
                    }
                )

            return json.dumps(formatted_rows, indent=2)
        except Exception as exc:
            logger.error("Oracle vector search failed: %s", exc)
            return ""

    def __del__(self) -> None:
        try:
            if getattr(self, "_owns_client", False) and getattr(self, "client", None):
                self.client.close()
        except Exception as exc:
            if "DPY-1001" not in str(exc):
                logger.error("Failed to close Oracle client: %s", exc)

        try:
            if getattr(self, "_openai_client", None):
                self._openai_client.close()
        except Exception as exc:
            logger.error("Failed to close OpenAI client: %s", exc)

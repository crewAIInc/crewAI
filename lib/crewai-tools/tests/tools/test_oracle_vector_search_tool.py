import importlib.util
import json
from decimal import Decimal
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import uuid

from crewai_tools import (
    OracleVectorSearchConfig,
    OracleVectorSearchQueryConfig,
    OracleVectorSearchTool,
)
import pytest

from crewai_tools.tools.oracle_vector_search_tool import vector_search as vs
from crewai_tools.tools.oracle_vector_search_tool.vector_search import _quote_identifier


ORACLE_USERNAME_ENV = "VECDB_USER"
ORACLE_PASS_ENV = "VECDB_PASS"
ORACLE_DSN_ENV = "VECDB_HOST"

class FakeOracleError(Exception):
    def __init__(self, code: int):
        super().__init__()
        self.args = [SimpleNamespace(code=code)]


class FakeCursor:
    def __init__(
        self,
        fetchall_result=None,
        fetchone_result=None,
        execute_side_effects=None,
    ):
        self.fetchall_result = fetchall_result or []
        self.fetchone_result = fetchone_result
        self.execute_side_effects = list(execute_side_effects or [])
        self.executed = []
        self.inputsizes = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def setinputsizes(self, *args):
        self.inputsizes = args

    def execute(self, sql, params=None, **kwargs):
        self.executed.append((sql, params, kwargs))
        if self.execute_side_effects:
            side_effect = self.execute_side_effects.pop(0)
            if side_effect is not None:
                raise side_effect

    def fetchall(self):
        return self.fetchall_result

    def fetchone(self):
        return self.fetchone_result


class FakeConnection:
    def __init__(self, cursors=None):
        self.cursors = list(cursors or [])
        self.commit_count = 0
        self.closed = False

    def cursor(self):
        return self.cursors.pop(0)

    def commit(self):
        self.commit_count += 1

    def close(self):
        self.closed = True


class FakePoolAcquire:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnectionPool:
    def __init__(self, connection):
        self.connection = connection
        self.acquire_count = 0
        self.closed = False

    def acquire(self):
        self.acquire_count += 1
        return FakePoolAcquire(self.connection)

    def close(self):
        self.closed = True


class FakeOracleModule:
    DB_TYPE_JSON = object()
    DB_TYPE_VECTOR = object()

    def __init__(self, connection=None):
        self.connection = connection
        self.connect_calls = []

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        return self.connection or FakeConnection()


class FakeEmbeddingsClient:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.closed = False

    def close(self):
        self.closed = True


class CloseError:
    def __init__(self, message: str):
        self.message = message

    def close(self):
        raise RuntimeError(self.message)


class FakeLob:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value


def _oracle_env_config() -> tuple[str, str, str] | None:
    username = os.getenv(ORACLE_USERNAME_ENV)
    password = os.getenv(ORACLE_PASS_ENV)
    dsn = os.getenv(ORACLE_DSN_ENV)
    if not username or not password or not dsn:
        return None
    return username, password, dsn


def _embed_text(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "oracle" in lowered else 0.0,
        1.0 if "crewai" in lowered else 0.0,
        1.0 if "vector" in lowered else 0.0,
    ]


def _cleanup_real_tool(tool: OracleVectorSearchTool) -> None:
    table_name = tool.oracle_config.table_name
    try:
        with tool.client.cursor() as cursor:
            cursor.execute(f'DROP TABLE "{table_name}" PURGE')
        tool.client.commit()
    except Exception:
        pass

    openai_client = getattr(tool, "_openai_client", None)
    if openai_client is not None:
        try:
            openai_client.close()
        except Exception:
            pass

    client = getattr(tool, "client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def make_tool(**kwargs):
    defaults = {
        "oracle_config": OracleVectorSearchConfig(table_name="docs_vectors"),
        "client": FakeConnection(),
        "embedding_function": lambda text: [float(len(text))],
        "dimensions": 1,
    }
    defaults.update(kwargs)
    return OracleVectorSearchTool(**defaults)


@pytest.fixture
def oracle_tool():
    creds = _oracle_env_config()
    if creds is not None:
        username, password, dsn = creds
        table_name = f'CREWAI_ORACLE_UNIT_{uuid.uuid4().hex[:12].upper()}'
        tool = OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(
                user=username,
                password=password,
                dsn=dsn,
                table_name=table_name,
                limit=2,
                distance_strategy="COSINE",
            ),
            embedding_function=_embed_text,
            dimensions=3,
        )
        tool.create_table()
        tool.add_texts(
            [
                "Oracle vector guide",
                "CrewAI onboarding",
            ],
            metadatas=[
                {"source": "docs", "priority": 5},
                {"source": "crew", "priority": 1},
            ],
        )
        try:
            yield tool
        finally:
            _cleanup_real_tool(tool)
        return

    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(
        fetchall_result=[
            ("CrewAI docs", '{"source": "docs"}', Decimal("0.12")),
        ]
    )
    client = FakeConnection([existence_cursor, search_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            table_name="docs_vectors",
            limit=2,
        ),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )
    yield tool


def test_successful_query_execution(oracle_tool):
    results = json.loads(
        oracle_tool._run(query="oracle vector search", filter_by="source", filter_value="docs")
    )

    assert len(results) >= 1
    assert results[0]["metadata"]["source"] == "docs"
    assert results[0]["context"]
    assert results[0]["score"] == pytest.approx(results[0]["distance"])

    if isinstance(oracle_tool.client, FakeConnection):
        assert results[0]["context"] == "CrewAI docs"
        assert results[0]["score"] == pytest.approx(0.12)
        assert results[0]["distance"] == pytest.approx(0.12)


def test_query_config_filters_results_by_score_threshold():
    creds = _oracle_env_config()
    if creds is not None:
        username, password, dsn = creds
        table_name = f'CREWAI_ORACLE_THRESH_{uuid.uuid4().hex[:12].upper()}'
        tool = OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(
                user=username,
                password=password,
                dsn=dsn,
                table_name=table_name,
                distance_strategy="COSINE",
            ),
            query_config=OracleVectorSearchQueryConfig(score_threshold=0.5),
            embedding_function=_embed_text,
            dimensions=3,
        )
        tool.create_table()
        tool.add_texts(
            ["Oracle vector guide", "CrewAI onboarding"],
            metadatas=[
                {"source": "docs"},
                {"source": "crew"},
            ],
        )
        try:
            results = json.loads(tool._run(query="oracle vector"))
            assert len(results) >= 1
            assert all(result["distance"] <= 0.5 for result in results)
            assert results[0]["metadata"]["source"] == "docs"
        finally:
            _cleanup_real_tool(tool)
        return

    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(
        fetchall_result=[
            ("CrewAI docs", '{"source": "docs"}', 0.1),
            ("Low relevance", '{"source": "other"}', 10.0),
        ]
    )
    client = FakeConnection([existence_cursor, search_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        query_config=OracleVectorSearchQueryConfig(score_threshold=0.5),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    results = json.loads(tool._run(query="oracle vector search"))

    assert len(results) == 1
    assert results[0]["metadata"]["source"] == "docs"
    assert results[0]["distance"] <= 0.5


def test_simple_filter_preserves_native_json_numeric_type():
    creds = _oracle_env_config()
    if creds is not None:
        username, password, dsn = creds
        table_name = f'CREWAI_ORACLE_FILTER_{uuid.uuid4().hex[:12].upper()}'
        tool = OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(
                user=username,
                password=password,
                dsn=dsn,
                table_name=table_name,
                distance_strategy="COSINE",
            ),
            embedding_function=_embed_text,
            dimensions=3,
        )
        tool.create_table()
        tool.add_texts(
            ["Oracle vector guide", "CrewAI onboarding"],
            metadatas=[
                {"source": "docs", "priority": 3},
                {"source": "crew", "priority": 1},
            ],
        )
        try:
            results = json.loads(
                tool._run(
                    query="oracle vector search",
                    filter_by="priority",
                    filter_value=3,
                )
            )
            assert len(results) >= 1
            assert all(result["metadata"]["priority"] == 3 for result in results)
        finally:
            _cleanup_real_tool(tool)
        return

    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(fetchall_result=[])
    client = FakeConnection([existence_cursor, search_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    tool._run(query="oracle vector search", filter_by="priority", filter_value=3)

    executed_sql, params, _ = search_cursor.executed[0]
    assert "JSON_EXISTS(metadata, '$.priority?(@ == $val)'" in executed_sql
    assert params["value0"] == 3


def test_run_normalizes_decimal_metadata_values():
    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(
        fetchall_result=[
            ("CrewAI docs", {"priority": Decimal("5"), "ratio": Decimal("0.25")}, 0.1),
        ]
    )
    client = FakeConnection([existence_cursor, search_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    results = json.loads(tool._run(query="oracle vector search"))

    assert results[0]["metadata"]["priority"] == 5
    assert results[0]["metadata"]["ratio"] == pytest.approx(0.25)


def test_per_call_score_threshold_filters_by_max_distance():
    creds = _oracle_env_config()
    if creds is not None:
        username, password, dsn = creds
        table_name = f'CREWAI_ORACLE_CALLTH_{uuid.uuid4().hex[:12].upper()}'
        tool = OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(
                user=username,
                password=password,
                dsn=dsn,
                table_name=table_name,
                distance_strategy="COSINE",
            ),
            embedding_function=_embed_text,
            dimensions=3,
        )
        tool.create_table()
        tool.add_texts(
            ["Oracle vector guide", "CrewAI onboarding"],
            metadatas=[
                {"source": "docs"},
                {"source": "crew"},
            ],
        )
        try:
            results = json.loads(
                tool._run(query="oracle vector search", score_threshold=0.5)
            )
            assert len(results) >= 1
            assert all(result["distance"] <= 0.5 for result in results)
            assert results[0]["metadata"]["source"] == "docs"
        finally:
            _cleanup_real_tool(tool)
        return

    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(
        fetchall_result=[
            ("Near match", '{"source": "docs"}', 0.2),
            ("Far match", '{"source": "docs"}', 0.8),
        ]
    )
    client = FakeConnection([existence_cursor, search_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    results = json.loads(tool._run(query="oracle vector search", score_threshold=0.5))

    assert [result["context"] for result in results] == ["Near match"]
    assert results[0]["score"] == pytest.approx(0.2)


def test_filters_json_builds_oracle_where_clause():
    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(fetchall_result=[])
    client = FakeConnection([existence_cursor, search_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    tool._run(
        query="oracle vector search",
        filters=json.dumps(
            {
                "$or": [
                    {"source": "docs"},
                    {"priority": {"$gte": 3}},
                ]
            }
        ),
    )

    executed_sql, params, _ = search_cursor.executed[0]
    assert "JSON_EXISTS(metadata, '$.source?(@ == $val)'" in executed_sql
    assert "JSON_EXISTS(metadata, '$.priority?(@ >= $val" in executed_sql
    assert 3 in params.values()


def test_create_vector_index(oracle_tool):
    index_cursor = FakeCursor()
    oracle_tool.client = FakeConnection([index_cursor])

    index_name = oracle_tool.create_vector_index(
        index_name="docs_vec_idx",
        params={"accuracy": 80, "neighbors": 64, "efconstruction": 300, "parallel": 4},
    )

    assert index_name == "docs_vec_idx"
    assert oracle_tool.client.commit_count == 1
    ddl = index_cursor.executed[0][0]
    assert "CREATE VECTOR INDEX" in ddl
    assert "ORGANIZATION INMEMORY NEIGHBOR GRAPH" in ddl
    assert "WITH TARGET ACCURACY 80" in ddl
    assert "PARAMETERS (type HNSW, neighbors 64, efconstruction 300)" in ddl
    assert "PARALLEL 4" in ddl


def test_create_vector_index_creates_ivf_index():
    index_cursor = FakeCursor()
    client = FakeConnection([index_cursor])
    tool = make_tool(client=client)

    index_name = tool.create_vector_index(
        index_name="docs_ivf_idx",
        idx_type="IVF",
        params={
            "accuracy": 70,
            "neighbor_partitions": 128,
            "samples_per_partition": 10,
            "min_vectors_per_partition": 0,
            "parallel": 2,
        },
    )

    assert index_name == "docs_ivf_idx"
    assert client.commit_count == 1
    ddl = index_cursor.executed[0][0]
    assert "CREATE VECTOR INDEX" in ddl
    assert "ORGANIZATION NEIGHBOR PARTITIONS" in ddl
    assert "WITH TARGET ACCURACY 70" in ddl
    assert (
        "PARAMETERS (type IVF, neighbor partitions 128, "
        "samples_per_partition 10, min_vectors_per_partition 0)"
    ) in ddl
    assert "PARALLEL 2" in ddl


@pytest.mark.parametrize(
    ("idx_type", "params", "message"),
    [
        ("HNSW", {"accuracy": 0}, "accuracy must be at least 1"),
        ("HNSW", {"accuracy": 101}, "accuracy must be at most 100"),
        ("HNSW", {"accuracy": "90"}, "accuracy must be an integer"),
        ("HNSW", {"neighbors": 1}, "neighbors must be at least 2"),
        ("HNSW", {"neighbors": 2049}, "neighbors must be at most 2048"),
        ("HNSW", {"neighbors": "32"}, "neighbors must be an integer"),
        ("HNSW", {"efconstruction": 0}, "efconstruction must be at least 1"),
        ("HNSW", {"efconstruction": 65536}, "efconstruction must be at most 65535"),
        ("HNSW", {"efconstruction": "200"}, "efconstruction must be an integer"),
        ("HNSW", {"parallel": 0}, "parallel must be at least 1"),
        ("HNSW", {"parallel": "8 NOLOGGING"}, "parallel must be an integer"),
        ("HNSW", {"parallel": True}, "parallel must be an integer"),
        ("HNSW", {"neighbor_partitions": 32}, "Invalid parameter: neighbor_partitions"),
        ("HNSW", {"idx_type": "HNSW"}, "Invalid parameter: idx_type"),
        ("IVF", {"accuracy": 0}, "accuracy must be at least 1"),
        ("IVF", {"accuracy": 101}, "accuracy must be at most 100"),
        ("IVF", {"neighbor_partitions": 0}, "neighbor_partitions must be at least 1"),
        (
            "IVF",
            {"neighbor_partitions": 10000001},
            "neighbor_partitions must be at most 10000000",
        ),
        (
            "IVF",
            {"neighbor_partitions": "32"},
            "neighbor_partitions must be an integer",
        ),
        (
            "IVF",
            {"samples_per_partition": 0},
            "samples_per_partition must be at least 1",
        ),
        (
            "IVF",
            {"samples_per_partition": "10"},
            "samples_per_partition must be an integer",
        ),
        (
            "IVF",
            {"min_vectors_per_partition": -1},
            "min_vectors_per_partition must be at least 0",
        ),
        (
            "IVF",
            {"min_vectors_per_partition": "0"},
            "min_vectors_per_partition must be an integer",
        ),
        ("IVF", {"parallel": 0}, "parallel must be at least 1"),
        ("IVF", {"neighbors": 32}, "Invalid parameter: neighbors"),
        ("IVF", {"idx_type": "IVF"}, "Invalid parameter: idx_type"),
    ],
)
def test_create_vector_index_validates_parameters(monkeypatch, idx_type, params, message):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)
    index_cursor = FakeCursor()
    client = FakeConnection([index_cursor])
    tool = make_tool(client=client)

    with pytest.raises(ValueError, match=message):
        tool.create_vector_index(idx_type=idx_type, params=params)

    assert index_cursor.executed == []
    assert client.commit_count == 0


def test_create_vector_index_rejects_invalid_idx_type(monkeypatch):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)
    tool = make_tool(client=FakeConnection([FakeCursor()]))

    with pytest.raises(ValueError, match="idx_type must be HNSW or IVF"):
        tool.create_vector_index(idx_type="FLAT")

    with pytest.raises(ValueError, match="idx_type must be HNSW or IVF"):
        tool.create_vector_index(idx_type=1)


def test_create_vector_index_rejects_non_dict_params(monkeypatch):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)
    index_cursor = FakeCursor()
    client = FakeConnection([index_cursor])
    tool = make_tool(client=client)

    with pytest.raises(ValueError, match="params must be a dictionary"):
        tool.create_vector_index(params=[])

    assert index_cursor.executed == []
    assert client.commit_count == 0


def test_table_and_index_exists_helpers():
    table_cursor = FakeCursor()
    index_cursor = FakeCursor(fetchone_result=("DOCS_VEC_IDX",))
    client = FakeConnection([table_cursor, index_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            table_name="docs_vectors",
            index_name="DOCS_VEC_IDX",
        ),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    assert tool.table_exists() is True
    assert tool.vector_index_exists() is True


def test_add_texts_creates_table_and_inserts_rows(monkeypatch):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    existence_cursor = FakeCursor(execute_side_effects=[FakeOracleError(942)])
    create_table_cursor = FakeCursor()
    insert_cursor = FakeCursor()
    client = FakeConnection([existence_cursor, create_table_cursor, insert_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
        dimensions=3,
    )

    ids = tool.add_texts(["hello"], metadatas=[{"source": "docs"}], ids=["abc"])

    assert ids == ["abc"]
    assert client.commit_count == 2
    assert insert_cursor.inputsizes[-1] is FakeOracleModule.DB_TYPE_VECTOR
    assert "INSERT INTO" in insert_cursor.executed[0][0]


def test_run_returns_empty_string_on_error():
    missing_table_cursor = FakeCursor(execute_side_effects=[FakeOracleError(942)])
    client = FakeConnection([missing_table_cursor])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="missing_vectors"),
        client=client,
        embedding_function=lambda _: [0.1, 0.2, 0.3],
    )

    assert tool._run(query="oracle") == ""


def test_quote_identifier_preserves_caller_casing():
    assert _quote_identifier("docs_vectors") == '"docs_vectors"'
    assert _quote_identifier("MySchema.docs_vectors") == '"MySchema"."docs_vectors"'


def test_module_handles_missing_oracledb_import(monkeypatch):
    import builtins

    module_path = Path(vs.__file__)
    spec = importlib.util.spec_from_file_location(
        "vector_search_missing_oracledb", module_path
    )
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "oracledb":
            raise ImportError("missing oracledb")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.ORACLEDB_AVAILABLE is False
    assert module.oracledb is Any


def test_helper_functions_cover_error_paths():
    assert vs._read_lob_if_needed(FakeLob("hello")) == "hello"
    assert vs._normalize_json_value([{"a": vs.Decimal("1.5")}]) == [{"a": 1.5}]

    with pytest.raises(ValueError, match="Identifier"):
        vs._quote_identifier("bad-name")

    assert vs._quote_identifier('"Mixed".plain') == '"Mixed"."plain"'

    with pytest.raises(ValueError, match="Invalid metadata key"):
        vs._validate_metadata_key("bad-key!")


def test_get_comparison_string_covers_all_branches(monkeypatch):
    bind_variables = []

    with pytest.raises(ValueError, match="Invalid operator"):
        vs._get_comparison_string("$bad", 1, bind_variables)

    with pytest.raises(ValueError, match="exactly 2 elements"):
        vs._get_comparison_string("$between", [1], bind_variables)

    with pytest.raises(ValueError, match="At least one bound"):
        vs._get_comparison_string("$between", [None, None], bind_variables)

    assert vs._get_comparison_string("$between", [1, None], bind_variables) == (
        "@ >= $val0",
        ':value0 AS "val0"',
    )
    assert vs._get_comparison_string("$between", [None, 3], bind_variables) == (
        "@ <= $val1",
        ':value1 AS "val1"',
    )

    with pytest.raises(ValueError, match="must be a non-empty list"):
        vs._get_comparison_string("$in", "not-a-list", bind_variables)

    assert (
        vs._get_comparison_string("$all", [1, 2], bind_variables)[0]
        == "@ == $val2 && @ == $val3"
    )
    assert (
        vs._get_comparison_string("$in", [4, 5], bind_variables)[0]
        == "@ in ($val4,$val5)"
    )

    monkeypatch.setitem(vs.COMPARISON_MAP, "$custom_empty", "")
    try:
        with pytest.raises(ValueError, match=r"Invalid operator: \$custom_empty\."):
            vs._get_comparison_string("$custom_empty", 1, bind_variables)
    finally:
        del vs.COMPARISON_MAP["$custom_empty"]


def test_generate_condition_covers_dict_branches():
    bind_variables = []
    condition = vs._generate_condition(
        "metadata_field",
        {
            "$gt": 1,
            "$lt": 9,
            "$not": {"$eq": "blocked"},
            "$exists": False,
            "$nin": [10, 11],
            "$eq": {"a": 1},
            "$ne": {"a": 2},
        },
        bind_variables,
    )

    assert condition.startswith("(")
    assert (
        "JSON_EXISTS(metadata, '$.metadata_field?(@ > $val0 && @ < $val1)'"
        in condition
    )
    assert "NOT (JSON_EXISTS(metadata, '$.metadata_field?(@ == $val2)'" in condition
    assert "NOT (JSON_EXISTS(metadata, '$.metadata_field'))" in condition
    assert (
        "JSON_EQUAL(JSON_QUERY(metadata, '$.metadata_field'), JSON(:value5))"
        in condition
    )
    assert (
        "NOT (JSON_EQUAL(JSON_QUERY(metadata, '$.metadata_field'), JSON(:value6)))"
        in condition
    )
    assert bind_variables[5] == '{"a": 1}'
    assert bind_variables[6] == '{"a": 2}'


def test_generate_condition_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="Nested metadata objects"):
        vs._generate_condition("field", {"nested": 1}, [])

    with pytest.raises(ValueError, match="must be a boolean"):
        vs._generate_condition("field", {"$exists": "yes"}, [])

    with pytest.raises(ValueError, match="Filter format is invalid"):
        vs._generate_condition("field", [1, 2], [])

    assert (
        vs._generate_condition("field", {"$exists": True}, [])
        == "JSON_EXISTS(metadata, '$.field')"
    )


def test_generate_where_clause_covers_validation_and_logical_branches():
    with pytest.raises(ValueError, match="Must be a dictionary"):
        vs._generate_where_clause([], [])

    with pytest.raises(ValueError, match="not a recognized logical operator"):
        vs._generate_where_clause({"$xor": [{"a": 1}]}, [])

    with pytest.raises(ValueError, match="require an array"):
        vs._generate_where_clause({"$or": {"a": 1}}, [])

    bind_variables = []
    condition = vs._generate_where_clause(
        {
            "$nor": [{"source": "docs"}, {"priority": {"$gte": 3}}],
            "tenant": "crew",
        },
        bind_variables,
    )

    assert condition.startswith("(")
    assert "( NOT (" in condition
    assert "JSON_EXISTS(metadata, '$.source?(@ == $val)'" in condition
    assert "JSON_EXISTS(metadata, '$.priority?(@ >= $val1)'" in condition
    assert "JSON_EXISTS(metadata, '$.tenant?(@ == $val)'" in condition


def test_build_metadata_filter_covers_errors_and_combined_filters():
    with pytest.raises(ValueError, match="provided together"):
        vs._build_metadata_filter("source", None)

    with pytest.raises(
        ValueError, match="letters, numbers, underscores, and dots only"
    ):
        vs._build_metadata_filter("source-name", "docs")

    clause, params = vs._build_metadata_filter(
        "priority",
        3,
        {"source": "docs"},
    )

    assert "JSON_EXISTS(metadata, '$.priority?(@ == $val)'" in clause
    assert "JSON_EXISTS(metadata, '$.source?(@ == $val)'" in clause
    assert params == {"value0": 3, "value1": "docs"}


def test_extract_oracle_error_code_handles_missing_code():
    assert vs._extract_oracle_error_code(FakeOracleError(942)) == 942
    assert vs._extract_oracle_error_code(Exception("boom")) is None


def test_setup_oracle_handles_missing_dependency_decline(monkeypatch):
    click_module = ModuleType("click")
    click_module.confirm = lambda message: False
    monkeypatch.setitem(sys.modules, "click", click_module)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", False)

    with pytest.raises(ImportError, match="Please install it"):
        OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
            client=FakeConnection(),
            embedding_function=lambda _: [1.0],
            dimensions=1,
        )


def test_setup_oracle_handles_missing_dependency_accept(monkeypatch):
    original_oracledb = vs.oracledb
    fake_oracledb = FakeOracleModule()
    click_module = ModuleType("click")
    click_module.confirm = lambda message: True
    subprocess_module = ModuleType("subprocess")
    calls = []
    subprocess_module.run = lambda cmd, check: calls.append((cmd, check))
    monkeypatch.setitem(sys.modules, "click", click_module)
    monkeypatch.setitem(sys.modules, "subprocess", subprocess_module)
    monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", False)

    try:
        tool = OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
            client=FakeConnection(),
            embedding_function=lambda _: [1.0],
            dimensions=1,
        )

        assert calls == [(["uv", "add", "oracledb"], True)]
        assert vs.oracledb is fake_oracledb
        assert vs.ORACLEDB_AVAILABLE is True
    finally:
        vs.oracledb = original_oracledb


def test_setup_oracle_rejects_string_embedding_function():
    with pytest.raises(ValueError, match="callable"):
        OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
            client=FakeConnection(),
            embedding_function="custom.embed",
            dimensions=1,
        )


def test_setup_oracle_delegates_empty_connection_params_to_oracledb(monkeypatch):
    fake_oracledb = FakeOracleModule(connection=FakeConnection())
    monkeypatch.setattr(vs, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)

    OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    assert fake_oracledb.connect_calls == [{}]


def test_setup_oracle_connects_when_credentials_are_present(monkeypatch):
    fake_oracledb = FakeOracleModule(connection=FakeConnection())
    monkeypatch.setattr(vs, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)

    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            user="user",
            password="pass",
            dsn="dsn",
            table_name="docs_vectors",
        ),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    assert tool._owns_client is True
    assert fake_oracledb.connect_calls == [
        {"user": "user", "password": "pass", "dsn": "dsn"}
    ]


def test_setup_oracle_allows_dsn_only_connection(monkeypatch):
    fake_oracledb = FakeOracleModule(connection=FakeConnection())
    monkeypatch.setattr(vs, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)

    OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            dsn="dsn",
            table_name="docs_vectors",
        ),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    assert fake_oracledb.connect_calls == [{"dsn": "dsn"}]


def test_setup_oracle_passes_connection_kwargs(monkeypatch):
    fake_oracledb = FakeOracleModule(connection=FakeConnection())
    monkeypatch.setattr(vs, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)

    OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            user="user",
            password="pass",
            dsn="dsn",
            table_name="docs_vectors",
            connection_kwargs={
                "config_dir": "/wallet",
                "wallet_location": "/wallet",
                "retry_count": 3,
            },
        ),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    assert fake_oracledb.connect_calls == [
        {
            "config_dir": "/wallet",
            "wallet_location": "/wallet",
            "retry_count": 3,
            "user": "user",
            "password": "pass",
            "dsn": "dsn",
        }
    ]


def test_setup_oracle_allows_connection_kwargs_only(monkeypatch):
    fake_oracledb = FakeOracleModule(connection=FakeConnection())
    monkeypatch.setattr(vs, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)

    OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            table_name="docs_vectors",
            connection_kwargs={
                "config_dir": "/wallet",
                "wallet_location": "/wallet",
            },
        ),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    assert fake_oracledb.connect_calls == [
        {
            "config_dir": "/wallet",
            "wallet_location": "/wallet",
        }
    ]


def test_setup_oracle_primary_connection_fields_override_kwargs(monkeypatch):
    fake_oracledb = FakeOracleModule(connection=FakeConnection())
    monkeypatch.setattr(vs, "oracledb", fake_oracledb)
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)

    OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            user="user",
            password="pass",
            dsn="dsn",
            table_name="docs_vectors",
            connection_kwargs={
                "user": "other",
                "password": "other",
                "dsn": "other",
            },
        ),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    assert fake_oracledb.connect_calls == [
        {"user": "user", "password": "pass", "dsn": "dsn"}
    ]


def test_create_vector_index_uses_connection_pool(monkeypatch):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)
    index_cursor = FakeCursor()
    connection = FakeConnection([index_cursor])
    pool = FakeConnectionPool(connection)
    tool = make_tool(client=pool)

    index_name = tool.create_vector_index(index_name="docs_vec_idx")

    assert index_name == "docs_vec_idx"
    assert pool.acquire_count == 1
    assert connection.commit_count == 1
    assert "CREATE VECTOR INDEX" in index_cursor.executed[0][0]


def test_create_vector_index_uses_ivf_default_name(monkeypatch):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)
    index_cursor = FakeCursor()
    client = FakeConnection([index_cursor])
    tool = make_tool(client=client)

    index_name = tool.create_vector_index(idx_type="IVF")

    assert index_name == "docs_vectors_IVF_IDX"
    assert "ORGANIZATION NEIGHBOR PARTITIONS" in index_cursor.executed[0][0]


def test_caller_supplied_connection_pool_is_not_closed(monkeypatch):
    monkeypatch.setattr(vs, "oracledb", FakeOracleModule())
    monkeypatch.setattr(vs, "ORACLEDB_AVAILABLE", True)
    pool = FakeConnectionPool(FakeConnection())
    tool = make_tool(client=pool)

    tool.__del__()

    assert pool.closed is False


def test_setup_oracle_uses_azure_openai_when_configured(monkeypatch):
    fake_openai_client = FakeEmbeddingsClient(embeddings=None)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(vs, "AzureOpenAI", lambda: fake_openai_client)

    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=FakeConnection(),
        dimensions=1,
    )

    assert tool._openai_client is fake_openai_client


def test_setup_oracle_uses_openai_client_when_api_key_present(monkeypatch):
    fake_openai_client = FakeEmbeddingsClient(embeddings=None)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(vs, "Client", lambda: fake_openai_client)

    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=FakeConnection(),
        dimensions=1,
    )

    assert tool._openai_client is fake_openai_client


def test_setup_oracle_requires_embedding_source(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
        OracleVectorSearchTool(
            oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
            client=FakeConnection(),
            dimensions=1,
        )


def test_embed_texts_uses_openai_client():
    embeddings = SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2]),
                SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )
    )
    tool = make_tool()
    tool.embedding_function = None
    tool._openai_client = FakeEmbeddingsClient(embeddings=embeddings)
    tool.dimensions = 2

    assert tool._embed_texts(["a", "bb"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_table_exists_reraises_non_942_errors():
    cursor = FakeCursor(execute_side_effects=[FakeOracleError(1111)])
    tool = make_tool(client=FakeConnection([cursor]))

    with pytest.raises(FakeOracleError):
        tool._table_exists('"docs_vectors"')


def test_index_exists_without_table_name():
    cursor = FakeCursor(fetchone_result=("IDX",))
    tool = make_tool(client=FakeConnection([cursor]))

    assert tool._index_exists('"IDX"') is True
    assert cursor.executed[0][2] == {"idx_name": "IDX"}


def test_vector_index_exists_uses_default_name():
    cursor = FakeCursor(fetchone_result=("DOCS_VECTORS_HNSW_IDX",))
    tool = make_tool(client=FakeConnection([cursor]))

    assert tool.vector_index_exists() is True


def test_create_table_returns_early_when_table_exists():
    tool = make_tool(client=FakeConnection())
    object.__setattr__(tool, "_table_exists", lambda table_name: True)

    tool.create_table()

    assert tool.client.commit_count == 0


def test_create_vector_index_uses_default_name():
    cursor = FakeCursor()
    tool = make_tool(client=FakeConnection([cursor]))

    index_name = tool.create_vector_index()

    assert index_name == "docs_vectors_HNSW_IDX"
    assert "CREATE VECTOR INDEX" in cursor.executed[0][0]


def test_add_texts_covers_validation_errors():
    tool = make_tool()

    assert tool.add_texts([]) == []

    with pytest.raises(ValueError, match="metadatas must match"):
        tool.add_texts(["a"], metadatas=[{}, {}])

    with pytest.raises(ValueError, match="ids must match"):
        tool.add_texts(["a"], ids=["id-1", "id-2"])

    object.__setattr__(tool, "create_table", lambda: None)
    tool.embedding_function = lambda _: [1.0, 2.0]
    tool.dimensions = 1
    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        tool.add_texts(["a"])


def test_run_merges_default_filter_and_rejects_non_object_filters():
    existence_cursor = FakeCursor()
    search_cursor = FakeCursor(fetchall_result=[])
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        query_config=vs.OracleVectorSearchQueryConfig(filter={"tenant": "crew"}),
        client=FakeConnection([existence_cursor, search_cursor]),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )

    tool._run(query="oracle", filters='{"source":"docs"}')
    executed_sql, params, _ = search_cursor.executed[0]
    assert "JSON_EXISTS(metadata, '$.tenant?(@ == $val)'" in executed_sql
    assert "JSON_EXISTS(metadata, '$.source?(@ == $val)'" in executed_sql
    assert params["value0"] == "crew"
    assert params["value1"] == "docs"

    failing_tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(table_name="docs_vectors"),
        client=FakeConnection([FakeCursor()]),
        embedding_function=lambda _: [1.0],
        dimensions=1,
    )
    assert failing_tool._run(query="oracle", filters="[]") == ""


def test_del_logs_cleanup_errors(caplog):
    tool = make_tool(client=CloseError("client close failed"))
    tool._owns_client = True
    tool._openai_client = CloseError("openai close failed")

    with caplog.at_level("ERROR"):
        tool.__del__()

    assert "Failed to close Oracle client" in caplog.text
    assert "Failed to close OpenAI client" in caplog.text

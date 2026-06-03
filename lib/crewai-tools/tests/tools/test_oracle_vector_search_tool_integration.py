import json
import os
import uuid

from crewai_tools import OracleVectorSearchConfig, OracleVectorSearchTool
import pytest


ORACLE_USERNAME_ENV = "VECDB_USER"
ORACLE_PASSWORD_ENV = "VECDB_PASS"
ORACLE_DSN_ENV = "VECDB_HOST"


def _oracle_env_config() -> tuple[str, str, str] | None:
    username = os.getenv(ORACLE_USERNAME_ENV)
    password = os.getenv(ORACLE_PASSWORD_ENV)
    dsn = os.getenv(ORACLE_DSN_ENV)
    if not username or not password or not dsn:
        return None
    return username, password, dsn


def test_oracle_env_config_requires_explicit_credentials(monkeypatch):
    monkeypatch.delenv(ORACLE_USERNAME_ENV, raising=False)
    monkeypatch.delenv(ORACLE_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(ORACLE_DSN_ENV, raising=False)

    assert _oracle_env_config() is None

    monkeypatch.setenv(ORACLE_USERNAME_ENV, "user")
    monkeypatch.setenv(ORACLE_PASSWORD_ENV, "password")
    assert _oracle_env_config() is None

    monkeypatch.setenv(ORACLE_DSN_ENV, "host/service")
    assert _oracle_env_config() == ("user", "password", "host/service")


def _embed_text(text: str) -> list[float]:
    lowered = text.lower()
    return [
        1.0 if "oracle" in lowered else 0.0,
        1.0 if "crewai" in lowered else 0.0,
        1.0 if "vector" in lowered else 0.0,
    ]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if isinstance(texts, str):
        raise TypeError("batch embedding requires a list of texts")
    return [_embed_text(text) for text in texts]


def _drop_table(tool: OracleVectorSearchTool, table_name: str) -> None:
    try:
        with tool.client.cursor() as cursor:
            cursor.execute(f'DROP TABLE "{table_name}" PURGE')
        tool.client.commit()
    except Exception:
        pass


@pytest.mark.timeout(120)
def test_oracle_vector_search_tool_with_real_connection(
    pytestconfig: pytest.Config,
) -> None:
    creds = _oracle_env_config()
    if creds is None:
        pytest.skip(
            f"Set {ORACLE_USERNAME_ENV}, {ORACLE_PASSWORD_ENV}, and {ORACLE_DSN_ENV} to run Oracle integration tests."
        )

    if getattr(pytestconfig.option, "block_network", False):
        pytest.skip(
            "Network access is blocked by pytest addopts. Re-run this test without --block-network to use a real Oracle connection."
        )

    username, password, dsn = creds
    oracledb = pytest.importorskip("oracledb")
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception as exc:
        pytest.skip(f"Database connection failed: {exc}")
    else:
        connection.close()

    table_name = f'CREWAI_ORACLE_TOOL_{uuid.uuid4().hex[:12].upper()}'

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

    try:
        tool.create_table()
        inserted_ids = tool.add_texts(
            [
                "CrewAI integrates with Oracle vector search.",
                "This unrelated text is about cooking pasta.",
            ],
            metadatas=[
                {"source": "docs", "topic": "oracle", "priority": 5},
                {"source": "kitchen", "topic": "food", "priority": 1},
            ],
        )

        assert len(inserted_ids) == 2
        assert tool.table_exists() is True
        assert tool.vector_index_exists() is False

        unfiltered_results = json.loads(tool._run(query="oracle vector"))
        assert len(unfiltered_results) >= 1
        assert unfiltered_results[0]["metadata"]["source"] == "docs"
        assert "Oracle vector search" in unfiltered_results[0]["context"]
        assert unfiltered_results[0]["score"] == pytest.approx(
            unfiltered_results[0]["distance"]
        )

        filtered_results = json.loads(
            tool._run(
                query="oracle vector",
                filter_by="source",
                filter_value="docs",
            )
        )
        assert len(filtered_results) >= 1
        assert all(row["metadata"]["source"] == "docs" for row in filtered_results)

        numeric_filter_results = json.loads(
            tool._run(
                query="oracle vector",
                filter_by="priority",
                filter_value=5,
            )
        )
        assert len(numeric_filter_results) >= 1
        assert all(result["metadata"]["priority"] == 5 for result in numeric_filter_results)

        json_filter_results = json.loads(
            tool._run(
                query="oracle vector",
                filters=json.dumps(
                    {
                        "$or": [
                            {"source": "docs"},
                            {"topic": {"$eq": "oracle"}},
                        ]
                    }
                ),
                score_threshold=0.5,
            )
        )
        assert len(json_filter_results) >= 1
        assert all(result["distance"] <= 0.5 for result in json_filter_results)

        created_index_name = tool.create_vector_index()
        assert created_index_name
        assert tool.vector_index_exists() is True
    finally:
        _drop_table(tool, table_name)

        openai_client = getattr(tool, "_openai_client", None)
        if openai_client is not None:
            openai_client.close()

        client = getattr(tool, "client", None)
        if client is not None:
            client.close()


@pytest.mark.timeout(120)
def test_oracle_vector_search_tool_with_real_connection_batch_embedding(
    pytestconfig: pytest.Config,
) -> None:
    creds = _oracle_env_config()
    if creds is None:
        pytest.skip(
            f"Set {ORACLE_USERNAME_ENV}, {ORACLE_PASSWORD_ENV}, and {ORACLE_DSN_ENV} to run Oracle integration tests."
        )

    if getattr(pytestconfig.option, "block_network", False):
        pytest.skip(
            "Network access is blocked by pytest addopts. Re-run this test without --block-network to use a real Oracle connection."
        )

    username, password, dsn = creds
    oracledb = pytest.importorskip("oracledb")
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception as exc:
        pytest.skip(f"Database connection failed: {exc}")
    else:
        connection.close()

    table_name = f'CREWAI_ORACLE_BATCH_{uuid.uuid4().hex[:12].upper()}'
    tool = OracleVectorSearchTool(
        oracle_config=OracleVectorSearchConfig(
            user=username,
            password=password,
            dsn=dsn,
            table_name=table_name,
            limit=2,
            distance_strategy="COSINE",
        ),
        embedding_function=_embed_texts,
        dimensions=3,
    )

    try:
        tool.create_table()
        tool.add_texts(
            [
                "CrewAI integrates with Oracle vector search.",
                "This unrelated text is about cooking pasta.",
            ],
            metadatas=[
                {"source": "docs"},
                {"source": "kitchen"},
            ],
        )

        results = json.loads(tool._run(query="oracle vector"))
        assert len(results) >= 1
        assert results[0]["metadata"]["source"] == "docs"
    finally:
        _drop_table(tool, table_name)
        client = getattr(tool, "client", None)
        if client is not None:
            client.close()


@pytest.mark.parametrize(
    ("idx_type", "params", "expected_parameter_text"),
    [
        ("HNSW", {"neighbors": 32, "efconstruction": 200}, "HNSW"),
        (
            "IVF",
            {
                "neighbor_partitions": 32,
                "samples_per_partition": 1,
                "min_vectors_per_partition": 0,
            },
            "IVF",
        ),
    ],
)
@pytest.mark.timeout(120)
def test_oracle_vector_search_tool_creates_vector_index_type(
    pytestconfig: pytest.Config,
    idx_type: str,
    params: dict[str, int],
    expected_parameter_text: str,
) -> None:
    creds = _oracle_env_config()
    if creds is None:
        pytest.skip(
            f"Set {ORACLE_USERNAME_ENV}, {ORACLE_PASSWORD_ENV}, and {ORACLE_DSN_ENV} to run Oracle integration tests."
        )

    if getattr(pytestconfig.option, "block_network", False):
        pytest.skip(
            "Network access is blocked by pytest addopts. Re-run this test without --block-network to use a real Oracle connection."
        )

    username, password, dsn = creds
    oracledb = pytest.importorskip("oracledb")
    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
    except Exception as exc:
        pytest.skip(f"Database connection failed: {exc}")
    else:
        connection.close()

    table_name = f'CREWAI_ORACLE_{idx_type}_{uuid.uuid4().hex[:12].upper()}'
    index_name = f'{table_name}_IDX'
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

    try:
        tool.create_table()
        tool.add_texts(["CrewAI Oracle vector index type test."])

        created_index_name = tool.create_vector_index(
            index_name=index_name,
            idx_type=idx_type,
            params=params,
        )

        assert created_index_name == index_name
        assert tool.vector_index_exists(index_name=index_name) is True
        with tool.client.cursor() as cursor:
            cursor.execute(
                """
                SELECT index_type, index_subtype
                FROM user_indexes
                WHERE index_name = :index_name
                """,
                index_name=index_name,
            )
            row = cursor.fetchone()
        assert row is not None
        assert row[0] == "VECTOR"
        assert expected_parameter_text in (row[1] or "").upper()
    finally:
        _drop_table(tool, table_name)
        client = getattr(tool, "client", None)
        if client is not None:
            client.close()

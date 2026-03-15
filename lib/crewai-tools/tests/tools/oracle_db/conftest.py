from __future__ import annotations

import array
import json
import os
import signal
from typing import Any
from unittest.mock import MagicMock
import uuid

import pytest

from crewai.rag.embeddings.factory import build_embedder


class CursorStub:
    def __init__(self, execute_side_effect=None):
        self.execute_side_effect = execute_side_effect
        self.description = []
        self._fetchall_result = []
        self._fetchone_result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, *args, **kwargs):
        if self.execute_side_effect:
            return self.execute_side_effect(self, sql, *args, **kwargs)
        return None

    def fetchall(self):
        return self._fetchall_result

    def fetchone(self):
        return self._fetchone_result


@pytest.fixture
def oracle_connection_mock() -> MagicMock:
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    return connection


@pytest.fixture
def oracle_live_config() -> dict[str, Any]:
    default_wallet_dir = None
    for candidate in (
        os.path.expanduser("~/.langchain-oracle-wallet"),
        os.path.expanduser("~/.oracle-wallet"),
    ):
        if os.path.exists(os.path.join(candidate, "tnsnames.ora")):
            default_wallet_dir = candidate
            break

    password = os.getenv(
        "ORACLE_DB_PASSWORD",
        os.getenv("ORACLE_PASSWORD"),
    )

    return {
        "user": os.getenv("ORACLE_DB_USER", os.getenv("ORACLE_USER", "ADMIN")),
        "password": password,
        "dsn": os.getenv("ORACLE_DB_DSN", os.getenv("ORACLE_DSN")),
        "config_dir": os.getenv("ORACLE_DB_CONFIG_DIR", default_wallet_dir),
        "wallet_location": os.getenv("ORACLE_DB_WALLET_LOCATION", default_wallet_dir),
        "wallet_password": os.getenv(
            "ORACLE_DB_WALLET_PASSWORD",
            os.getenv("ORACLE_WALLET_PASSWORD"),
        )
        or password,
    }


@pytest.fixture
def oracle_live_text_tool_kwargs(oracle_live_config: dict[str, Any]) -> dict[str, Any]:
    return {
        **oracle_live_config,
        "table_name": os.getenv("ORACLE_DB_TEXT_TABLE", ""),
        "text_column": os.getenv("ORACLE_DB_TEXT_COLUMN", "text"),
        "metadata_columns": [
            column.strip()
            for column in os.getenv("ORACLE_DB_TEXT_METADATA_COLUMNS", "").split(",")
            if column.strip()
        ],
    }


@pytest.fixture
def oracle_live_hybrid_tool_kwargs(
    oracle_live_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        **oracle_live_config,
        "hybrid_index_name": os.getenv("ORACLE_DB_HYBRID_INDEX_NAME", ""),
        "table_name": os.getenv("ORACLE_DB_HYBRID_TABLE", ""),
        "text_column": os.getenv("ORACLE_DB_HYBRID_TEXT_COLUMN", "text"),
        "metadata_column": os.getenv("ORACLE_DB_HYBRID_METADATA_COLUMN", "metadata"),
        "metadata_columns": [
            column.strip()
            for column in os.getenv("ORACLE_DB_HYBRID_METADATA_COLUMNS", "").split(",")
            if column.strip()
        ],
        "search_mode": os.getenv("ORACLE_DB_HYBRID_SEARCH_MODE", "hybrid"),
    }


def has_oracle_text_test_config() -> bool:
    return bool(
        oracle_live_defaults_available()
    )


def has_oracle_hybrid_test_config() -> bool:
    return bool(
        oracle_live_defaults_available()
    )


def has_oracle_vector_test_config() -> bool:
    return bool(oracle_live_defaults_available()) and bool(
        os.getenv("OCI_COMPARTMENT_ID")
        and (os.getenv("OCI_REGION") or os.getenv("OCI_SERVICE_ENDPOINT"))
    )


def oracle_live_defaults_available() -> bool:
    wallet_candidates = (
        os.path.expanduser("~/.oracle-wallet"),
        os.path.expanduser("~/.langchain-oracle-wallet"),
    )
    wallet_available = any(
        os.path.exists(os.path.join(candidate, "tnsnames.ora"))
        for candidate in wallet_candidates
    )
    return wallet_available


@pytest.fixture
def oracle_live_vector_tool_kwargs(oracle_live_config: dict[str, Any]) -> dict[str, Any]:
    embedding_model: dict[str, Any] = {
        "provider": "oci",
        "config": {
            "model_name": os.getenv("OCI_EMBED_MODEL_NAME", "cohere.embed-v4.0"),
            "compartment_id": os.getenv("OCI_COMPARTMENT_ID"),
            "auth_profile": os.getenv("OCI_AUTH_PROFILE", "DEFAULT"),
            "auth_file_location": os.getenv(
                "OCI_AUTH_FILE_LOCATION", os.path.expanduser("~/.oci/config")
            ),
            "output_dimensions": int(os.getenv("OCI_EMBED_OUTPUT_DIMENSIONS", "1536")),
        },
    }
    if os.getenv("OCI_REGION"):
        embedding_model["config"]["region"] = os.getenv("OCI_REGION")
    if os.getenv("OCI_SERVICE_ENDPOINT"):
        embedding_model["config"]["service_endpoint"] = os.getenv(
            "OCI_SERVICE_ENDPOINT"
        )

    return {
        **oracle_live_config,
        "text_column": "text",
        "embedding_column": "embedding",
        "metadata_column": "metadata",
        "metadata_columns": ["category"],
        "embedding_model": embedding_model,
    }


class _ConnectTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _ConnectTimeout("Oracle DB connect timed out")


@pytest.fixture
def oracle_live_connection(oracle_live_config: dict[str, Any]):
    try:
        import oracledb
    except ImportError:
        pytest.skip("oracledb is not installed")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(20)
    try:
        connection = oracledb.connect(**oracle_live_config)
    except _ConnectTimeout as exc:
        pytest.skip(str(exc))
    except Exception as exc:
        pytest.skip(f"Oracle DB connection failed: {type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    try:
        yield connection
    finally:
        try:
            connection.close()
        except Exception:
            pass


@pytest.fixture
def oracle_text_live_resources(oracle_live_connection):
    suffix = uuid.uuid4().hex[:8].upper()
    table_name = f"CT_ORTXT_{suffix}"
    index_name = f"CT_OIDX_{suffix}"

    with oracle_live_connection.cursor() as cursor:
        cursor.execute(
            f"CREATE TABLE {table_name} (text VARCHAR2(4000), category VARCHAR2(100))"
        )
        cursor.execute(
            f"INSERT INTO {table_name}(text, category) VALUES (:1, :2)",
            [
                "Our refund policy for premium plans allows refunds within 30 days.",
                "billing",
            ],
        )
        cursor.execute(
            f"INSERT INTO {table_name}(text, category) VALUES (:1, :2)",
            [
                "Autonomous Database is Oracle managed database infrastructure.",
                "database",
            ],
        )
        cursor.execute(f"CREATE SEARCH INDEX {index_name} ON {table_name}(text)")
        oracle_live_connection.commit()

    try:
        yield {
            "table_name": table_name,
            "text_column": "text",
            "metadata_columns": ["category"],
        }
    finally:
        with oracle_live_connection.cursor() as cursor:
            try:
                cursor.execute(f"DROP INDEX {index_name}")
            except Exception:
                pass
            try:
                cursor.execute(f"DROP TABLE {table_name} PURGE")
            except Exception:
                pass
            oracle_live_connection.commit()


@pytest.fixture
def oracle_hybrid_live_resources(oracle_live_connection):
    suffix = uuid.uuid4().hex[:8].upper()
    table_name = f"CT_ORHY_{suffix}"
    index_name = f"CT_OHYIDX_{suffix}"
    preference_name = f"CT_OPREF_{suffix}"

    with oracle_live_connection.cursor() as cursor:
        cursor.execute(
            f"CREATE TABLE {table_name} (text VARCHAR2(4000), metadata JSON, category VARCHAR2(100))"
        )
        cursor.execute(
            f"INSERT INTO {table_name}(text, metadata, category) VALUES (:1, :2, :3)",
            [
                "Oracle Autonomous Database provides managed database capabilities.",
                json.dumps({"source": "oracle-docs"}),
                "database",
            ],
        )
        cursor.execute(
            f"INSERT INTO {table_name}(text, metadata, category) VALUES (:1, :2, :3)",
            [
                "Premium refund policy allows refunds within 30 days.",
                json.dumps({"source": "policy-docs"}),
                "billing",
            ],
        )
        cursor.execute(
            """
            begin
              DBMS_VECTOR_CHAIN.CREATE_PREFERENCE(
                :pref_name,
                dbms_vector_chain.vectorizer,
                json(:pref_json)
              );
            end;
            """,
            {
                "pref_name": preference_name,
                "pref_json": '{"model":"allminilm"}',
            },
        )
        try:
            # Oracle Hybrid Vector Index creation uses the database-side vectorizer
            # registered in DBMS_VECTOR_CHAIN. That is distinct from OCI GenAI
            # embedding models available through the configured OCI profile.
            #
            # On Autonomous Database 26ai, this hybrid-index path depends on a
            # supported in-database embedding/vectorizer model being installed in
            # the database itself. If the requested model is unavailable, Oracle
            # raises ORA-40284 during index creation. That is a database capability
            # gap for this tenancy/database, not a CrewAI tool failure.
            #
            # References:
            # - https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/create-hybrid-vector-index.html
            # - https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/guidelines-and-restrictions-hybrid-vector-indexes.html
            # - https://docs.oracle.com/en/database/oracle/oracle-database/26/arpls/dbms_hybrid_vector1.html
            cursor.execute(
                f"CREATE HYBRID VECTOR INDEX {index_name} ON {table_name}(text) "
                f"PARAMETERS ('vectorizer {preference_name}')"
            )
        except Exception as exc:
            oracle_live_connection.rollback()
            if "ORA-40284" in str(exc):
                pytest.skip(
                    "Oracle DB hybrid vectorizer model is not available in this 26ai database; "
                    "hybrid indexes here require a supported DB-local vectorizer model rather "
                    "than OCI GenAI embedding models"
                )
            raise
        oracle_live_connection.commit()

    try:
        yield {
            "hybrid_index_name": index_name,
            "table_name": table_name,
            "text_column": "text",
            "metadata_column": "metadata",
            "metadata_columns": ["category"],
            "search_mode": "hybrid",
        }
    finally:
        with oracle_live_connection.cursor() as cursor:
            try:
                cursor.execute(f"DROP INDEX {index_name}")
            except Exception:
                pass
            try:
                cursor.execute(
                    """
                    begin
                      DBMS_VECTOR_CHAIN.DROP_PREFERENCE(:pref_name);
                    end;
                    """,
                    {"pref_name": preference_name},
                )
            except Exception:
                pass
            try:
                cursor.execute(f"DROP TABLE {table_name} PURGE")
            except Exception:
                pass
            oracle_live_connection.commit()


@pytest.fixture
def oracle_vector_live_resources(
    oracle_live_connection,
    oracle_live_vector_tool_kwargs: dict[str, Any],
):
    suffix = uuid.uuid4().hex[:8].upper()
    table_name = f"CT_ORVEC_{suffix}"
    output_dimensions = oracle_live_vector_tool_kwargs["embedding_model"]["config"][
        "output_dimensions"
    ]

    try:
        embedder = build_embedder(oracle_live_vector_tool_kwargs["embedding_model"])
        documents = [
            (
                "Premium refund policy allows refunds within 30 days of purchase.",
                {"source": "policy-docs", "topic": "billing"},
                "billing",
            ),
            (
                "Autonomous Database is Oracle managed database infrastructure.",
                {"source": "oracle-docs", "topic": "database"},
                "database",
            ),
            (
                "Vector search compares embeddings using similarity distance.",
                {"source": "vector-docs", "topic": "ai"},
                "ai",
            ),
        ]
        embeddings = embedder([document[0] for document in documents])
    except Exception as exc:
        pytest.skip(f"OCI embedding setup failed: {type(exc).__name__}: {exc}")

    with oracle_live_connection.cursor() as cursor:
        cursor.execute(
            f"CREATE TABLE {table_name} ("
            f"id VARCHAR2(64), "
            f"text VARCHAR2(4000), "
            f"metadata JSON, "
            f"category VARCHAR2(100), "
            f"embedding VECTOR({output_dimensions}, FLOAT32)"
            f")"
        )
        for index, ((text, metadata, category), embedding) in enumerate(
            zip(documents, embeddings, strict=True),
            start=1,
        ):
            cursor.execute(
                f"INSERT INTO {table_name}(id, text, metadata, category, embedding) "
                f"VALUES (:1, :2, :3, :4, :5)",
                [
                    f"doc-{index}",
                    text,
                    json.dumps(metadata),
                    category,
                    array.array("f", [float(value) for value in embedding]),
                ],
            )
        oracle_live_connection.commit()

    try:
        yield {
            "table_name": table_name,
            "text_column": "text",
            "embedding_column": "embedding",
            "metadata_column": "metadata",
            "metadata_columns": ["category"],
            "embedder": embedder,
        }
    finally:
        with oracle_live_connection.cursor() as cursor:
            try:
                cursor.execute(f"DROP TABLE {table_name} PURGE")
            except Exception:
                pass
            oracle_live_connection.commit()

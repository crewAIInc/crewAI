"""Live integration tests for Oracle embeddings with Crew memory."""

from __future__ import annotations

import os

import pytest

from crewai.memory.unified_memory import Memory
from crewai.rag.embeddings.factory import build_embedder


ORACLE_USERNAME_ENV = "VECDB_USER"
ORACLE_PASSWORD_ENV = "VECDB_PASS"
ORACLE_DSN_ENV = "VECDB_HOST"
ORACLE_MODEL_ENV = "ORACLE_EMBEDDING_MODEL"


def _oracle_env_config() -> tuple[str, str, str] | None:
    username = os.getenv(ORACLE_USERNAME_ENV, "")
    password = os.getenv(ORACLE_PASSWORD_ENV, "")
    dsn = os.getenv(ORACLE_DSN_ENV, "")
    if not username or not password or not dsn:
        return None
    return username, password, dsn


@pytest.mark.timeout(120)
def test_oracle_embedder_and_memory_with_real_connection(
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

    pytest.importorskip("oracledb")
    username, password, dsn = creds
    model_name = os.getenv(ORACLE_MODEL_ENV, "ALL_MINILM_L12_V2")
    embedder_config = {
        "provider": "oracle",
        "config": {
            "connection_params": {
                "user": username,
                "password": password,
                "dsn": dsn,
            },
            "embedding_params": {
                "provider": "database",
                "model": model_name,
            },
        },
    }

    embedder = build_embedder(embedder_config)
    vectors = embedder(["database tablespace", "kitchen recipes"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])

    memory = Memory(embedder=embedder_config)
    record = memory.remember(
        content="A tablespace can be online or offline whenever the database is open.",
        scope="/oracle",
        categories=["oracle", "storage"],
        importance=0.8,
    )
    assert record is not None
    assert record.scope == "/oracle"

    results = memory.recall(
        "tablespace online offline database",
        scope="/oracle",
        depth="shallow",
        limit=3,
    )
    assert results, "Expected shallow recall to return at least one Oracle-embedded memory"
    assert "tablespace" in results[0].record.content.lower()

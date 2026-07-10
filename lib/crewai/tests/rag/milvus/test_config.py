"""Tests for MilvusConfig."""

from crewai.rag.milvus.config import MilvusConfig


def test_milvus_config_defaults_to_lite_uri() -> None:
    """Test that Milvus config defaults to a local Milvus Lite URI."""
    config = MilvusConfig(embedding_function=lambda text: [0.1, 0.2, 0.3])

    assert config.provider == "milvus"
    assert config.options == {"uri": "./milvus.db"}
    assert config.dimension == 1536
    assert config.metric_type == "COSINE"


def test_milvus_config_accepts_server_options() -> None:
    """Test that Milvus config accepts server and cloud connection options."""
    config = MilvusConfig(
        options={
            "uri": "https://example.api.gcp-us-west1.zillizcloud.com",
            "token": "test-token",
            "db_name": "crew",
        },
        embedding_function=lambda text: [0.1, 0.2, 0.3],
        consistency_level="Bounded",
    )

    assert config.options["uri"].startswith("https://")
    assert config.options["token"] == "test-token"
    assert config.options["db_name"] == "crew"
    assert config.consistency_level == "Bounded"

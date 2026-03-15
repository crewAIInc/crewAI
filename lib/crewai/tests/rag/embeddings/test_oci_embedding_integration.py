from __future__ import annotations

from crewai.rag.embeddings.factory import build_embedder


def test_oci_live_embedding_call(
    oci_embeddings_live_config: dict[str, object],
    oci_embedding_provider_config,
) -> None:
    embedder = build_embedder(
        {
            "provider": "oci",
            "config": oci_embedding_provider_config(
                str(oci_embeddings_live_config["text_model_env"]),
                str(oci_embeddings_live_config["text_model_default"]),
            ),
        }
    )

    result = embedder(list(oci_embeddings_live_config["text_inputs"]))

    assert len(result) == 2
    assert all(len(embedding) > 0 for embedding in result)

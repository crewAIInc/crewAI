from __future__ import annotations

import pytest

from crewai.rag.embeddings.factory import build_embedder


def test_oci_live_image_embedding_call(
    oci_embeddings_live_config: dict[str, object],
    oci_embedding_provider_config,
) -> None:
    text_embedder = build_embedder(
        {
            "provider": "oci",
            "config": oci_embedding_provider_config(
                str(oci_embeddings_live_config["text_model_env"]),
                str(oci_embeddings_live_config["text_model_default"]),
            ),
        }
    )
    image_embedder = build_embedder(
        {
            "provider": "oci",
            "config": oci_embedding_provider_config(
                str(oci_embeddings_live_config["image_model_env"]),
                str(oci_embeddings_live_config["image_model_default"]),
            ),
        }
    )

    text_vector = text_embedder([str(oci_embeddings_live_config["image_query"])])[0]
    try:
        image_vector = image_embedder.embed_image(
            bytes(oci_embeddings_live_config["image_bytes"]),
            mime_type="image/png",
        )
    except Exception as exc:
        if "Entity with key" in str(exc) and "not found" in str(exc):
            pytest.skip(
                "OCI image embedding model is listed in this tenancy but not invokable via embedText."
            )
        raise

    assert len(text_vector) > 0
    assert len(image_vector) > 0

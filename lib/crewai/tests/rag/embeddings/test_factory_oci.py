"""Tests for OCI embedding provider wiring."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crewai.rag.embeddings.factory import build_embedder
from crewai.rag.embeddings.providers.oci.embedding_callable import OCIEmbeddingFunction


class _FakeOCI:
    def __init__(self) -> None:
        self.retry = SimpleNamespace(DEFAULT_RETRY_STRATEGY="retry")
        self.config = SimpleNamespace(
            from_file=lambda file_location, profile_name: {
                "file_location": file_location,
                "profile_name": profile_name,
            }
        )
        self.signer = SimpleNamespace(
            load_private_key_from_file=lambda *_args, **_kwargs: "private-key"
        )
        self.auth = SimpleNamespace(
            signers=SimpleNamespace(
                SecurityTokenSigner=lambda token, key: (token, key),
                InstancePrincipalsSecurityTokenSigner=lambda: "instance-principal",
                get_resource_principals_signer=lambda: "resource-principal",
            )
        )
        self.generative_ai_inference = SimpleNamespace(
            GenerativeAiInferenceClient=MagicMock(),
            models=SimpleNamespace(
                EmbedTextDetails=_simple_init_class("EmbedTextDetails"),
                OnDemandServingMode=_simple_init_class("OnDemandServingMode"),
                DedicatedServingMode=_simple_init_class("DedicatedServingMode"),
            ),
        )


def _simple_init_class(name: str):
    class _Simple:
        output_dimensions = None

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _Simple.__name__ = name
    return _Simple


@patch("crewai.rag.embeddings.factory.import_and_validate_definition")
def test_build_embedder_oci(mock_import):
    """Test building OCI embedder."""
    mock_provider_class = MagicMock()
    mock_provider_instance = MagicMock()
    mock_embedding_function = MagicMock()

    mock_import.return_value = mock_provider_class
    mock_provider_class.return_value = mock_provider_instance
    mock_provider_instance.embedding_callable.return_value = mock_embedding_function

    config = {
        "provider": "oci",
        "config": {
            "model_name": "cohere.embed-english-v3.0",
            "compartment_id": "ocid1.compartment.oc1..test",
            "region": "us-chicago-1",
            "auth_profile": "DEFAULT",
        },
    }

    build_embedder(config)

    mock_import.assert_called_once_with(
        "crewai.rag.embeddings.providers.oci.oci_provider.OCIProvider"
    )
    call_kwargs = mock_provider_class.call_args.kwargs
    assert call_kwargs["model_name"] == "cohere.embed-english-v3.0"
    assert call_kwargs["compartment_id"] == "ocid1.compartment.oc1..test"
    assert call_kwargs["region"] == "us-chicago-1"


def test_oci_embedding_function_batches_requests(monkeypatch):
    """Test OCI embedding batching and request construction."""
    fake_oci = _FakeOCI()
    fake_client = MagicMock()
    fake_client.embed_text.side_effect = [
        SimpleNamespace(data=SimpleNamespace(embeddings=[[0.1, 0.2], [0.3, 0.4]])),
        SimpleNamespace(data=SimpleNamespace(embeddings=[[0.5, 0.6]])),
    ]
    fake_oci.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    monkeypatch.setattr(
        "crewai.rag.embeddings.providers.oci.embedding_callable._get_oci_module",
        lambda: fake_oci,
    )

    embedder = OCIEmbeddingFunction(
        model_name="cohere.embed-english-v3.0",
        compartment_id="ocid1.compartment.oc1..test",
        region="us-chicago-1",
        batch_size=2,
    )

    result = embedder(["a", "b", "c"])

    result_rows = [embedding.tolist() for embedding in result]
    expected_rows = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert len(result_rows) == len(expected_rows)
    for actual, expected in zip(result_rows, expected_rows, strict=True):
        assert actual == pytest.approx(expected)
    assert fake_client.embed_text.call_count == 2
    first_request = fake_client.embed_text.call_args_list[0].args[0]
    assert first_request.compartment_id == "ocid1.compartment.oc1..test"
    assert first_request.serving_mode.model_id == "cohere.embed-english-v3.0"


def test_oci_embedding_function_supports_output_dimensions(monkeypatch):
    """Test OCI output_dimensions mapping."""
    fake_oci = _FakeOCI()
    fake_client = MagicMock()
    fake_client.embed_text.return_value = SimpleNamespace(
        data=SimpleNamespace(embeddings=[[0.1, 0.2]])
    )
    fake_oci.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    monkeypatch.setattr(
        "crewai.rag.embeddings.providers.oci.embedding_callable._get_oci_module",
        lambda: fake_oci,
    )

    embedder = OCIEmbeddingFunction(
        model_name="cohere.embed-v4.0",
        compartment_id="ocid1.compartment.oc1..test",
        output_dimensions=512,
    )

    embedder(["hello"])

    request = fake_client.embed_text.call_args.args[0]
    assert request.output_dimensions == 512


def test_oci_embedding_function_exposes_serializable_config(monkeypatch):
    """Test OCI embedding config serialization for ChromaDB compatibility."""
    fake_oci = _FakeOCI()
    fake_oci.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        MagicMock()
    )

    monkeypatch.setattr(
        "crewai.rag.embeddings.providers.oci.embedding_callable._get_oci_module",
        lambda: fake_oci,
    )

    embedder = OCIEmbeddingFunction(
        model_name="cohere.embed-english-v3.0",
        compartment_id="ocid1.compartment.oc1..test",
        timeout=(5, 30),
    )

    assert embedder.get_config() == {
        "model_name": "cohere.embed-english-v3.0",
        "compartment_id": "ocid1.compartment.oc1..test",
        "timeout": [5, 30],
    }

    rebuilt = OCIEmbeddingFunction.build_from_config(embedder.get_config())
    assert rebuilt.get_config() == embedder.get_config()


def test_oci_embedding_function_supports_image_embeddings(monkeypatch, tmp_path: Path):
    """Test OCI image embedding request construction."""
    fake_oci = _FakeOCI()
    fake_client = MagicMock()
    fake_client.embed_text.return_value = SimpleNamespace(
        data=SimpleNamespace(embeddings=[[0.7, 0.8, 0.9]])
    )
    fake_oci.generative_ai_inference.GenerativeAiInferenceClient.return_value = (
        fake_client
    )

    monkeypatch.setattr(
        "crewai.rag.embeddings.providers.oci.embedding_callable._get_oci_module",
        lambda: fake_oci,
    )

    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    embedder = OCIEmbeddingFunction(
        model_name="cohere.embed-v4.0",
        compartment_id="ocid1.compartment.oc1..test",
    )

    result = embedder.embed_image(image_path)

    assert result == pytest.approx([0.7, 0.8, 0.9])
    request = fake_client.embed_text.call_args.args[0]
    assert request.input_type == "IMAGE"
    assert request.inputs[0].startswith("data:image/png;base64,")

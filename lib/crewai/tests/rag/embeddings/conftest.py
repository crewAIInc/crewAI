from __future__ import annotations

import io
import os

from PIL import Image
import pytest


def _valid_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _has_oci_test_config() -> bool:
    return bool(
        os.getenv("OCI_COMPARTMENT_ID")
        and (os.getenv("OCI_SERVICE_ENDPOINT") or os.getenv("OCI_REGION"))
    )


def _has_oci_sdk() -> bool:
    try:
        import oci  # noqa: F401
    except ImportError:
        return False
    return True


def _embedding_provider_config(model_env_var: str, default_model: str) -> dict[str, object]:
    config: dict[str, object] = {
        "model_name": os.getenv(model_env_var, default_model),
        "compartment_id": os.getenv("OCI_COMPARTMENT_ID"),
        "auth_type": os.getenv("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.getenv("OCI_AUTH_PROFILE", "DEFAULT"),
        "auth_file_location": os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
    }
    if os.getenv("OCI_REGION"):
        config["region"] = os.getenv("OCI_REGION")
    if os.getenv("OCI_SERVICE_ENDPOINT"):
        config["service_endpoint"] = os.getenv("OCI_SERVICE_ENDPOINT")
    return config


@pytest.fixture
def oci_embeddings_live_config() -> dict[str, object]:
    if not _has_oci_sdk() or not _has_oci_test_config():
        pytest.skip(
            "Requires OCI SDK plus OCI_COMPARTMENT_ID and OCI endpoint configuration"
        )
    return {
        "text_model_env": "OCI_EMBED_TEST_MODEL",
        "text_model_default": "cohere.embed-english-v3.0",
        "image_model_env": "OCI_IMAGE_EMBED_TEST_MODEL",
        "image_model_default": "cohere.embed-v4.0",
        "text_inputs": [
            "Oracle Cloud Infrastructure provides cloud services.",
            "Autonomous Database is an Oracle managed database service.",
        ],
        "image_query": "OCI architecture diagram",
        "image_bytes": _valid_png_bytes(),
    }


@pytest.fixture
def oci_embedding_provider_config():
    return _embedding_provider_config


@pytest.fixture
def allowed_hosts() -> list[str]:
    return [r".*"]

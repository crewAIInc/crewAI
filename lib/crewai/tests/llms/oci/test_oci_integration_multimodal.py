"""Live integration tests for OCI Generative AI multimodal content.

Run with:
    OCI_AUTH_TYPE=API_KEY OCI_AUTH_PROFILE=API_KEY_AUTH \
    OCI_COMPARTMENT_ID=<compartment> OCI_REGION=us-chicago-1 \
    OCI_TEST_MULTIMODAL_MODELS="google.gemini-2.5-flash" \
    uv run pytest tests/llms/oci/test_oci_integration_multimodal.py -v
"""

from __future__ import annotations

import base64
import os
import struct
import zlib

import pytest

from crewai.llms.providers.oci.completion import OCICompletion


def _env_models(env_var: str, fallback: str, default: str) -> list[str]:
    raw = os.getenv(env_var) or os.getenv(fallback) or default
    return [m.strip() for m in raw.split(",") if m.strip()]


def _skip_unless_live():
    compartment = os.getenv("OCI_COMPARTMENT_ID")
    if not compartment:
        pytest.skip("OCI_COMPARTMENT_ID not set")
    region = os.getenv("OCI_REGION")
    endpoint = os.getenv("OCI_SERVICE_ENDPOINT")
    if not region and not endpoint:
        pytest.skip("Set OCI_REGION or OCI_SERVICE_ENDPOINT")
    config: dict[str, str] = {"compartment_id": compartment}
    if endpoint:
        config["service_endpoint"] = endpoint
    if os.getenv("OCI_AUTH_TYPE"):
        config["auth_type"] = os.getenv("OCI_AUTH_TYPE", "API_KEY")
    if os.getenv("OCI_AUTH_PROFILE"):
        config["auth_profile"] = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
    return config


def _make_red_png() -> bytes:
    """Generate a minimal valid 2x2 red PNG image in memory."""
    width, height = 2, 2
    # Each row: filter byte (0) + RGB pixels
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter: none
        for _ in range(width):
            raw_data += b"\xff\x00\x00"  # red pixel

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", zlib.compress(raw_data))
    png += _chunk(b"IEND", b"")
    return png


def _png_data_uri() -> str:
    """Return a data URI for a small red PNG."""
    png_bytes = _make_red_png()
    encoded = base64.standard_b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@pytest.fixture(
    params=_env_models(
        "OCI_TEST_MULTIMODAL_MODELS", "OCI_TEST_MULTIMODAL_MODEL", "google.gemini-2.5-flash"
    ),
    ids=lambda m: m,
)
def oci_multimodal_model(request):
    return request.param


@pytest.fixture()
def oci_multimodal_config():
    return _skip_unless_live()


def test_oci_live_image_input(oci_multimodal_model: str, oci_multimodal_config: dict):
    """Vision model should describe an image sent as a data URI."""
    llm = OCICompletion(model=oci_multimodal_model, **oci_multimodal_config)
    result = llm.call(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Answer in one word."},
                    {"type": "image_url", "image_url": {"url": _png_data_uri()}},
                ],
            }
        ]
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert "red" in result.lower()

from __future__ import annotations

import base64
import io
import os
import wave

from PIL import Image
import pytest


def _png_data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buffer, format="PNG")
    return (
        "data:image/png;base64,"
        f"{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    )


def _pdf_data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(buffer, format="PDF")
    return (
        "data:application/pdf;base64,"
        f"{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    )


def _wav_data_uri() -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)
    return (
        "data:audio/wav;base64,"
        f"{base64.b64encode(buffer.getvalue()).decode('ascii')}"
    )


def test_oci_live_image_input(
    oci_multimodal_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    llm = oci_live_llm_factory(
        oci_multimodal_model,
        max_tokens=oci_token_budget(oci_multimodal_model, "basic"),
        temperature=oci_temperature_for_model(oci_multimodal_model),
    )

    result = llm.call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with a short sentence about this image."},
                    {"type": "image_url", "image_url": {"url": _png_data_uri()}},
                ],
            }
        ]
    )

    assert isinstance(result, str)
    assert result.strip()


def test_oci_live_pdf_input(
    oci_multimodal_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    if not oci_multimodal_model.startswith("google.gemini"):
        pytest.skip("PDF multimodal coverage currently requires a Gemini OCI model")

    llm = oci_live_llm_factory(
        oci_multimodal_model,
        max_tokens=oci_token_budget(oci_multimodal_model, "basic"),
        temperature=oci_temperature_for_model(oci_multimodal_model),
    )

    result = llm.call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply briefly after inspecting this PDF."},
                    {"type": "document_url", "document_url": {"url": _pdf_data_uri()}},
                ],
            }
        ]
    )

    assert isinstance(result, str)
    assert result.strip()


def test_oci_live_audio_input(
    oci_multimodal_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    if not oci_multimodal_model.startswith("google.gemini"):
        pytest.skip("Audio multimodal coverage currently requires a Gemini OCI model")

    llm = oci_live_llm_factory(
        oci_multimodal_model,
        max_tokens=oci_token_budget(oci_multimodal_model, "basic"),
        temperature=oci_temperature_for_model(oci_multimodal_model),
    )

    result = llm.call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply briefly after inspecting this audio."},
                    {"type": "audio_url", "audio_url": {"url": _wav_data_uri()}},
                ],
            }
        ]
    )

    assert isinstance(result, str)
    assert result.strip()


def test_oci_live_video_input(
    oci_multimodal_model: str,
    oci_live_llm_factory,
    oci_temperature_for_model,
    oci_token_budget,
):
    if not oci_multimodal_model.startswith("google.gemini"):
        pytest.skip("Video multimodal coverage currently requires a Gemini OCI model")

    video_data_uri = os.getenv("OCI_TEST_VIDEO_DATA_URI")
    if not video_data_uri:
        pytest.skip("Configure OCI_TEST_VIDEO_DATA_URI for OCI live video tests")

    llm = oci_live_llm_factory(
        oci_multimodal_model,
        max_tokens=oci_token_budget(oci_multimodal_model, "basic"),
        temperature=oci_temperature_for_model(oci_multimodal_model),
    )

    result = llm.call(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply briefly after inspecting this video."},
                    {"type": "video_url", "video_url": {"url": video_data_uri}},
                ],
            }
        ]
    )

    assert isinstance(result, str)
    assert result.strip()

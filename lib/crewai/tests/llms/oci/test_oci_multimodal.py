"""Unit tests for OCI provider multimodal content (mocked SDK)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_oci_builds_image_content(patch_oci_module, oci_unit_values):
    """image_url content should produce ImageContent with ImageUrl."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    # Mock multimodal content types
    patch_oci_module.generative_ai_inference.models.ImageContent = MagicMock(
        side_effect=lambda **kw: MagicMock(type="image", **kw)
    )
    patch_oci_module.generative_ai_inference.models.ImageUrl = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    content = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
    ]
    result = llm._build_generic_content(content)

    assert len(result) == 2
    patch_oci_module.generative_ai_inference.models.ImageContent.assert_called_once()
    patch_oci_module.generative_ai_inference.models.ImageUrl.assert_called_once()


def test_oci_builds_document_content(patch_oci_module, oci_unit_values):
    """document_url content should produce DocumentContent."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    patch_oci_module.generative_ai_inference.models.DocumentContent = MagicMock(
        side_effect=lambda **kw: MagicMock(type="document", **kw)
    )
    patch_oci_module.generative_ai_inference.models.DocumentUrl = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    content = [{"type": "document_url", "document_url": {"url": "data:application/pdf;base64,xyz"}}]
    result = llm._build_generic_content(content)

    assert len(result) == 1
    patch_oci_module.generative_ai_inference.models.DocumentContent.assert_called_once()


def test_oci_builds_video_content(patch_oci_module, oci_unit_values):
    """video_url content should produce VideoContent."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    patch_oci_module.generative_ai_inference.models.VideoContent = MagicMock(
        side_effect=lambda **kw: MagicMock(type="video", **kw)
    )
    patch_oci_module.generative_ai_inference.models.VideoUrl = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    content = [{"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}]
    result = llm._build_generic_content(content)

    assert len(result) == 1
    patch_oci_module.generative_ai_inference.models.VideoContent.assert_called_once()


def test_oci_builds_audio_content(patch_oci_module, oci_unit_values):
    """audio_url content should produce AudioContent."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    patch_oci_module.generative_ai_inference.models.AudioContent = MagicMock(
        side_effect=lambda **kw: MagicMock(type="audio", **kw)
    )
    patch_oci_module.generative_ai_inference.models.AudioUrl = MagicMock(
        side_effect=lambda **kw: MagicMock(**kw)
    )

    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    content = [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,wav123"}}]
    result = llm._build_generic_content(content)

    assert len(result) == 1
    patch_oci_module.generative_ai_inference.models.AudioContent.assert_called_once()


def test_oci_rejects_unsupported_content_type(patch_oci_module, oci_unit_values):
    """Unknown content types should raise ValueError."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )

    with pytest.raises(ValueError, match="Unsupported OCI content type"):
        llm._build_generic_content([{"type": "hologram", "data": "xyz"}])


def test_oci_cohere_rejects_multimodal(patch_oci_module, oci_unit_values):
    """Cohere models should reject multimodal content in _build_chat_request."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["cohere_model"],
        compartment_id=oci_unit_values["compartment_id"],
    )

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}
    ]

    with pytest.raises(ValueError, match="text-only"):
        llm._build_chat_request(messages)


def test_oci_message_has_multimodal_content(patch_oci_module, oci_unit_values):
    """_message_has_multimodal_content should detect non-text content types."""
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )

    assert llm._message_has_multimodal_content("just text") is False
    assert llm._message_has_multimodal_content([{"type": "text", "text": "hi"}]) is False
    assert llm._message_has_multimodal_content([{"type": "image_url", "image_url": {"url": "x"}}]) is True
    assert llm._message_has_multimodal_content([{"type": "text"}, {"type": "audio_url"}]) is True


def test_oci_supports_multimodal(patch_oci_module, oci_unit_values):
    from crewai.llms.providers.oci.completion import OCICompletion

    patch_oci_module.generative_ai_inference.GenerativeAiInferenceClient.return_value = MagicMock()
    llm = OCICompletion(
        model=oci_unit_values["model"],
        compartment_id=oci_unit_values["compartment_id"],
    )
    assert llm.supports_multimodal() is True


def test_vision_helpers():
    """Test vision.py utility functions."""
    from crewai.llms.providers.oci.vision import (
        VISION_MODELS,
        encode_image,
        is_vision_model,
        to_data_uri,
    )

    # to_data_uri with bytes
    uri = to_data_uri(b"\x89PNG", "image/png")
    assert uri.startswith("data:image/png;base64,")

    # to_data_uri passthrough
    existing = "data:image/jpeg;base64,abc"
    assert to_data_uri(existing) == existing

    # encode_image
    result = encode_image(b"\x89PNG")
    assert result["type"] == "image_url"
    assert result["image_url"]["url"].startswith("data:image/png;base64,")

    # is_vision_model
    assert is_vision_model("google.gemini-2.5-flash") is True
    assert is_vision_model("meta.llama-3.3-70b-instruct") is False

    assert len(VISION_MODELS) > 0

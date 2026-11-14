"""Regression tests for OpenAI Responses file constraints."""

from unittest.mock import patch

from crewai_files import AudioFile, format_multimodal_content
from crewai_files.core.sources import FileBytes
from crewai_files.processing.constraints import get_supported_content_types


def _mp3() -> AudioFile:
    with patch(
        "crewai_files.core.sources.detect_content_type", return_value="audio/mpeg"
    ):
        return AudioFile(
            source=FileBytes(
                data=b"ID3\x04\x00\x00\x00\x00\x00\x00payload", filename="clip.mp3"
            )
        )


def test_responses_does_not_advertise_audio_support():
    """Audio is not a supported Responses input, so it must not be advertised."""
    assert "audio/" not in get_supported_content_types("openai", api="responses")


def test_audio_input_is_skipped_instead_of_raising():
    """Audio files fall through to tools instead of crashing the formatter."""
    blocks = format_multimodal_content({"clip": _mp3()}, "openai", api="responses")

    assert blocks == []

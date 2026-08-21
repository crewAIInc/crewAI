"""Tests for the file uploader factory."""

from crewai_files.processing.exceptions import PermanentUploadError
from crewai_files.uploaders.factory import get_uploader
import pytest


def test_get_uploader_raises_for_unconfigured_bedrock(monkeypatch):
    monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)

    with pytest.raises(
        PermanentUploadError, match="Bedrock S3 uploader not configured"
    ):
        get_uploader("bedrock")


def test_get_uploader_raises_for_unsupported_provider():
    with pytest.raises(
        PermanentUploadError, match="No file uploader available for provider"
    ):
        get_uploader("totally-not-a-real-provider")

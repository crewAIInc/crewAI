"""Tests for the uploader factory function get_uploader."""


from crewai_files.processing.exceptions import PermanentUploadError
from crewai_files.uploaders.anthropic import AnthropicFileUploader
from crewai_files.uploaders.factory import get_uploader
from crewai_files.uploaders.gemini import GeminiFileUploader
from crewai_files.uploaders.openai import OpenAIFileUploader
import pytest


class TestGetUploader:
    """Tests for get_uploader factory function."""

    def test_get_uploader_anthropic(self):
        """Test get_uploader returns AnthropicFileUploader for 'anthropic'."""
        uploader = get_uploader("anthropic")
        assert isinstance(uploader, AnthropicFileUploader)

    def test_get_uploader_gemini(self):
        """Test get_uploader returns GeminiFileUploader for 'gemini'."""
        uploader = get_uploader("gemini")
        assert isinstance(uploader, GeminiFileUploader)

    def test_get_uploader_openai(self):
        """Test get_uploader returns OpenAIFileUploader for 'openai'."""
        uploader = get_uploader("openai")
        assert isinstance(uploader, OpenAIFileUploader)

    def test_get_uploader_unsupported_provider_raises(self):
        """Test that an unsupported provider raises PermanentUploadError, not RuntimeError."""
        with pytest.raises(PermanentUploadError, match="No file uploader available"):
            get_uploader("invalid_provider")

    def test_get_uploader_bedrock_without_config_raises(self, monkeypatch):
        """Test that bedrock provider without S3 bucket config raises PermanentUploadError."""
        # Ensure the env var is not set
        monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)

        with pytest.raises(PermanentUploadError, match="Bedrock S3 uploader not configured"):
            get_uploader("bedrock")

    def test_get_uploader_bedrock_with_bucket_env(self, monkeypatch):
        """Test that bedrock provider works when CREWAI_BEDROCK_S3_BUCKET is set."""
        monkeypatch.setenv("CREWAI_BEDROCK_S3_BUCKET", "my-test-bucket")

        # Lazy import so the test doesn't fail if boto3 isn't installed
        try:
            from crewai_files.uploaders.bedrock import BedrockFileUploader
        except ImportError:
            pytest.skip("boto3 not installed, skipping bedrock uploader test")

        uploader = get_uploader("bedrock")
        assert isinstance(uploader, BedrockFileUploader)

    def test_get_uploader_bedrock_with_bucket_kwarg(self, monkeypatch):
        """Test that bedrock provider works when bucket_name is passed as kwarg."""
        # Ensure env var is NOT set, so we only rely on the kwarg
        monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)

        try:
            from crewai_files.uploaders.bedrock import BedrockFileUploader
        except ImportError:
            pytest.skip("boto3 not installed, skipping bedrock uploader test")

        uploader = get_uploader("bedrock", bucket_name="my-kwarg-bucket")
        assert isinstance(uploader, BedrockFileUploader)
